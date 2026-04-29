import reed_solomon_ccsds as rs
from pycsp import Packet, HMACEngine, CRCEngine
from typing import Union
import socket
try:
    import pyserial
except:
    pyserial = None

class Golay24:
    N = 12
    H = [
        0x8008ED, 0x4001DB, 0x2003B5, 0x100769,
        0x080ED1, 0x040DA3, 0x020B47, 0x01068F,
        0x008D1D, 0x004A3B, 0x002477, 0x001FFE,
    ]
    
    @staticmethod  
    def __parity(x: int) -> int:
        """
        Return the parity (0 if even number of 1‑bits, 1 if odd).
        """
        return x.bit_count() & 1

    @classmethod     
    def encode(cls, r: int) -> int:
        """
        Encode a 12‑bit word into a 24‑bit Golay codeword.
    
        Args:
            data: integer whose lower 12 bits are the information word (0 ≤ data < 4096).
    
        Returns:
            A 24‑bit integer: (12 parity bits << 12) | (12 data bits).
        """
        assert r < 4096, 'data must be 0..4095'
        
        # Compute the 12‑bit syndrome/parity
        s = 0
        for h in cls.H:
            # Shift left to make room for the next parity bit
            s <<= 1
            # XOR in the parity of (H[i] & r)
            s |= cls.__parity(h & r)
    
        # Assemble codeword: parity bits in high half, data bits in low half
        codeword = ((s & 0xFFF) << cls.N) | r
        return codeword

    @classmethod  
    def decode(cls, codeword: int) -> tuple[int, int]:
        """
        Decode a 24‑bit Golay codeword.
    
        Args:
            codeword: 24‑bit integer (parity<<12 | data)
    
        Returns:
            (corrected_codeword, error_count), where error_count = number of flipped bits
            If uncorrectable, returns (original_codeword, -1).
        """
        r = codeword
        # Step 1: compute 12‑bit syndrome s = H * r
        s = 0
        for h in cls.H:
            s = (s << 1) | cls.__parity(h & r)
    
        # Step 2: if wt(s) ≤ 3, error vector e = (s << 12)
        if s.bit_count() <= 3:
            e = s << cls.N
            return (r ^ e, e.bit_count())
    
        # Step 3: for each i, if wt(s ^ B(i)) ≤ 2, e = ((s ^ B(i)) << 12) | (1 << (11‑i))
        for i in range(cls.N):
            b = cls.H[i] & 0xFFF  # B(i)
            if (s ^ b).bit_count() <= 2:
                e = ((s ^ b) << cls.N) | (1 << (cls.N - i - 1))
                return (r ^ e, e.bit_count())
    
        # Step 4: compute modified syndrome q = B * s
        q = 0
        for i in range(cls.N):
            b = cls.H[i] & 0xFFF
            q = (q << 1) | cls.__parity(b & s)
    
        # Step 5: if wt(q) ≤ 3, e = q
        if q.bit_count() <= 3:
            e = q
            return (r ^ e, e.bit_count())
    
        # Step 6: for each i, if wt(q ^ B(i)) ≤ 2, e = (1 << (2*12 - i - 1)) | (q ^ B(i))
        for i in range(cls.N):
            b = cls.H[i] & 0xFFF
            if (q ^ b).bit_count() <= 2:
                e = (1 << (2*cls.N - i - 1)) | (q ^ b)
                return (r ^ e, e.bit_count())
    
        # Step 7: uncorrectable
        return (codeword, -1)

class CCSDSRxScrambler:
    """
    CCSDS RX scrambler:
      • h(x) = x^8 + x^7 + x^5 + x^3 + 1
      • order = 8, fbmask = 0xA9, initreg = 0xFF
      • uses a 256‑byte precomputed table
    """
    _TABLE = bytes([
        0xFF, 0x48, 0x0E, 0xC0, 0x9A, 0x0D, 0x70, 0xBC,
        0x8E, 0x2C, 0x93, 0xAD, 0xA7, 0xB7, 0x46, 0xCE,
        0x5A, 0x97, 0x7D, 0xCC, 0x32, 0xA2, 0xBF, 0x3E,
        0x0A, 0x10, 0xF1, 0x88, 0x94, 0xCD, 0xEA, 0xB1,
        0xFE, 0x90, 0x1D, 0x81, 0x34, 0x1A, 0xE1, 0x79,
        0x1C, 0x59, 0x27, 0x5B, 0x4F, 0x6E, 0x8D, 0x9C,
        0xB5, 0x2E, 0xFB, 0x98, 0x65, 0x45, 0x7E, 0x7C,
        0x14, 0x21, 0xE3, 0x11, 0x29, 0x9B, 0xD5, 0x63,
        0xFD, 0x20, 0x3B, 0x02, 0x68, 0x35, 0xC2, 0xF2,
        0x38, 0xB2, 0x4E, 0xB6, 0x9E, 0xDD, 0x1B, 0x39,
        0x6A, 0x5D, 0xF7, 0x30, 0xCA, 0x8A, 0xFC, 0xF8,
        0x28, 0x43, 0xC6, 0x22, 0x53, 0x37, 0xAA, 0xC7,
        0xFA, 0x40, 0x76, 0x04, 0xD0, 0x6B, 0x85, 0xE4,
        0x71, 0x64, 0x9D, 0x6D, 0x3D, 0xBA, 0x36, 0x72,
        0xD4, 0xBB, 0xEE, 0x61, 0x95, 0x15, 0xF9, 0xF0,
        0x50, 0x87, 0x8C, 0x44, 0xA6, 0x6F, 0x55, 0x8F,
        0xF4, 0x80, 0xEC, 0x09, 0xA0, 0xD7, 0x0B, 0xC8,
        0xE2, 0xC9, 0x3A, 0xDA, 0x7B, 0x74, 0x6C, 0xE5,
        0xA9, 0x77, 0xDC, 0xC3, 0x2A, 0x2B, 0xF3, 0xE0,
        0xA1, 0x0F, 0x18, 0x89, 0x4C, 0xDE, 0xAB, 0x1F,
        0xE9, 0x01, 0xD8, 0x13, 0x41, 0xAE, 0x17, 0x91,
        0xC5, 0x92, 0x75, 0xB4, 0xF6, 0xE8, 0xD9, 0xCB,
        0x52, 0xEF, 0xB9, 0x86, 0x54, 0x57, 0xE7, 0xC1,
        0x42, 0x1E, 0x31, 0x12, 0x99, 0xBD, 0x56, 0x3F,
        0xD2, 0x03, 0xB0, 0x26, 0x83, 0x5C, 0x2F, 0x23,
        0x8B, 0x24, 0xEB, 0x69, 0xED, 0xD1, 0xB3, 0x96,
        0xA5, 0xDF, 0x73, 0x0C, 0xA8, 0xAF, 0xCF, 0x82,
        0x84, 0x3C, 0x62, 0x25, 0x33, 0x7A, 0xAC, 0x7F,
        0xA4, 0x07, 0x60, 0x4D, 0x06, 0xB8, 0x5E, 0x47,
        0x16, 0x49, 0xD6, 0xD3, 0xDB, 0xA3, 0x67, 0x2D,
        0x4B, 0xBE, 0xE6, 0x19, 0x51, 0x5F, 0x9F, 0x05,
        0x08, 0x78, 0xC4, 0x4A, 0x66, 0xF5, 0x58,
    ])

    def __init__(self, skip:int=0):
        """
        :param skip: number of initial bytes to leave unscrambled
        """
        self.skip = skip

    def __call__(self, data:Union[bytes, bytearray, memoryview]) -> bytes:
        """
        Scramble (descramble) the input data according to the CCSDS RX table.
        First `skip` bytes are passed through; the rest are XOR’d with the table.
        :param data: input buffer
        :return: scrambled output as bytes
        """
        tbl = self._TABLE
        tlen = len(tbl)
        out = bytearray(len(data))
        for i in range(self.skip, len(data)):
            out[i] = data[i] ^ tbl[(i - self.skip) % tlen]
        return out

class AX100:
    ASM = b'\x93\x0b\x51\xde'
    
    def __init__(self, hmac_key:bytes=None, crc=False, reed_solomon=False, randomize=True, len_field=True, syncword=True, prefill=32, tailfill=1, exception=False, verbose=False):
        self.hmac_engine = HMACEngine(hmac_key) if not hmac_key is None else None
        self.crc_engine = CRCEngine() if crc else None
        self.reed_solomon = reed_solomon
        self.scrambler = CCSDSRxScrambler() if randomize else None
        self.len_field = len_field
        self.syncword = syncword
        self.prefill = prefill
        self.tailfill = tailfill
        self.exception = exception
        self.verbose = verbose

    def encode(self, packet:Union[Packet, bytes, bytearray, memoryview]) -> bytes:
        if isinstance(packet, Packet):
            x = packet.encode()
        else:
            x = packet

        if self.hmac_engine:
            x = x + self.hmac_engine(x)

        if self.crc_engine:
            x = x + self.crc_engine(x)

        if self.reed_solomon:
            padding = 0
            if len(x) > 223: 
                x = x[:223]
            else:
                padding = 223 - len(x)
                x = bytes(padding) + x

            coded = rs.encode(x, False, 1)
            x = coded[padding:]

        if self.scrambler:
            x = self.scrambler(x)

        if self.len_field:
            golay = Golay24.encode(len(x)).to_bytes(3, 'big')
            x = golay + x

        if self.syncword:
            x = self.ASM + x
        
        return self.prefill*b'\xaa' + x + self.tailfill*b'\xaa'

    def decode(self, data:Union[bytes, bytearray, memoryview]) -> Packet|None:
        if self.syncword:
            if self.verbose: 
                if data[0:4] != self.ASM: print('ASM ERROR')
            data = data[4:]

        if self.len_field:
            pkt_len, errcnt = Golay24.decode(int.from_bytes(data[0:3], 'big'))
            pkt_len &= 0xfff
            if errcnt < 0: 
                if self.exception: raise ValueError('GOLAY ERROR')
                return None

            data = data[3:3+pkt_len]

        if self.scrambler:  # descramble here 
            data = self.scrambler(data)

        if self.reed_solomon:
            if len(data) < 32:
                if self.verbose: print('packet too short')
                if self.exception: raise ValueError('packet too short')
                return None
            
            if len(data) > 255:
                data = data[255:]
            else:
                padding = 255 - len(data)
                data = bytes(padding) + data

            try:
                errs, decoded = rs.decode(data, False, 1)
                if self.verbose and errs[0] != 0:
                    print('RS CORR=%d' % errs[0])
            except rs.UncorrectableError:
                if self.verbose: print('RS ERROR')
                if self.exception: raise ValueError('RS ERROR')
                return None
                
            data = decoded[padding:]

        if self.crc_engine:
            crc_val = data[-4:]
            if len(crc_val) != 4:
                if self.verbose: print('packet too short')
                if self.exception: raise ValueError('packet too short')
                return None

            if crc_val != self.crc_engine(data[:-4]):
                if self.verbose: print('CRC ERROR')
                if self.exception: raise ValueError('CRC ERROR')
                return None
                
            data = data[:-4]
        
        if self.hmac_engine:
            hmac_val = data[-4:]
            if len(hmac_val) != 4:
                if self.verbose: print('packet too short')
                if self.exception: raise ValueError('packet too short')
                return None
                    
            if hmac_val != self.hmac_engine(data[:-4]):
                if self.verbose: print('HMAC ERROR')
                if self.exception: raise ValueError('HMAC ERROR')
                return None
            
            data = data[:-4]
        
        packet = Packet()
        packet.decode(data)
        return packet

class KISS:
    def __init__(self):
        pass

class GrcLink:
    def __init__(self, host='127.0.0.1', port=52001, mtu=1024, timeout=1):
        self.s = socket.create_connection((host, port))
        self.mtu = mtu
        self.timeout = timeout
        self.s.settimeout(timeout)

    def __del__(self):
        self.close()
    
    def send(self, raw_data, data):
        self.s.sendall(raw_data + data)

    def recv(self):
        return self.s.recv(self.mtu)

    def close(self):
        self.s.close()
class Interface:
    def __init__(self, name='', mtu=256, timeout=1):
        self.mtu = mtu
        self.timeout = timeout
        self.name = name
        
    def send(self, pkt:Packet):
        pass

    def recv(self, timeout=None) -> Packet|None:
        return None

class Loopback(Interface):
    def __init__(self, name='lo', mtu=65536, timeout=1, queue_limit=1024):
        super().__init__(name, mtu, timeout)
        self.queue = []
        self.queue_limit = queue_limit
    
    def send(self, pkt:Packet):
        if len(self.queue) >= self.queue_limit:
            self.queue.pop(0)
        self.queue.append(pkt)
    
    def recv(self, timeout=None):
        try:
            return self.queue.pop()
        except:
            return None

def GrcAX100(Interface):
    def __init__(self, name='radio', remote='127.0.0.1', port=52001, mtu=256, timeout=1):
        pass

def Tcp(Interface):
    def __init__(self, name='tcp', remote='127.0.0.1', port=52001, listen=None, max_clients=0, mtu=65536, timeout=1):
        '''
        use listen='0.0.0.0' for tcp server mode
        '''
        pass

def Udp(Interface):
    def __init__(self, name='udp', 
                 listen='127.0.0.1', port=2612, 
                 remote=None, remote_port=2612, 
                 mtu=65507, timeout=1):
        pass

def SerialKISS(Interface):
    def __init__(self, name='serial', dev='/dev/ttyUSB0', baud=115200, mtu=256, timeout=1):
        '''
        use dev='tcp://127.0.0.1:2620' for tcp client mode
        '''
        pass
