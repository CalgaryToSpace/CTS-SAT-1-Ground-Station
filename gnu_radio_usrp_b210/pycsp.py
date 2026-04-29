import xtea, random, hashlib, crc
from typing import Literal, Union

class HeaderV1:
    '''
    CSP 1.x header 
     31         29         24         19      13      7      3                       0
    +----------+----------+----------+-------+-------+------+------+------+-----+-----+ 
    | Priority | Src Addr | Dst Addr | dport | sport | RSVD | HMAC | XTEA | RDP | CRC |
    | (2b)     | (5b)     | (5b)     | (6b)  | (6b)  | (4b) | (1)  | (1)  | (1) | (1) |
    +----------+----------+----------+-------+-------+------+------+------+-----+-----+ 
    Big endian is the standard implementation
    '''
    __slots__ = (
        'src', 'dst', 'dport', 'sport', 'prio', 'flags', 'endian'
    )

    PRIO_CRITICAL		= 0
    PRIO_HIGH			= 1
    PRIO_NORM			= 2
    PRIO_LOW			= 3

    FLAG_HMAC = 8
    FLAG_XTEA = 4
    FLAG_RDP = 2
    FLAG_CRC = 1
    
    def __init__(self, src:int, dst: int, dport: int, sport: int,
                 prio:int|Literal['critical', 'high', 'norm', 'low']='norm',
                 flags:int=0, 
                 endian:Literal['big', 'little']='big',
                 hmac:bool=None, 
                 xtea:bool=None, 
                 rdp:bool=None, 
                 crc:bool=None):
        # Validate fields
        assert 0 <= src   <= 31, 'src addr must be 0..31'
        assert 0 <= dst   <= 31, 'dst addr must be 0..31'
        assert 0 <= dport <= 63, 'dst port must be 0..63' 
        assert 0 <= sport <= 63, 'src port must be 0..63'
        if isinstance(prio, int):
            assert 0 <= prio    <= 3 , 'priority must be 0..3'
        else:
            prio = ['critical', 'high', 'norm', 'low'].index(prio)
        assert 0 <= flags <= 0xff, 'flags must be 0x00..0xff'
            
        self.src = src
        self.dst = dst
        self.dport = dport
        self.sport = sport
        
        self.prio = prio
        self.flags = flags
        
        if not hmac is None: self.hmac = hmac
        if not xtea is None: self.xtea = xtea
        if not rdp is None:  self.rdp = rdp
        if not crc is None:  self.crc = crc
        
        self.endian = endian

    @classmethod
    def from_bytes(cls, b:bytes, endian:str='big') -> 'CSP.HeaderV1':
        '''
        Parse a 4-byte CSP header (V1)
        '''
        assert len(b) == 4, 'CSP Header V1 must have exactly 4 bytes'
        h = int.from_bytes(b, byteorder=endian)
        
        # extract in order from MSB
        prio  = (h >> 30) & 0x3
        src   = (h >> 25) & 0x1f
        dst   = (h >> 20) & 0x1f
        dport = (h >> 14) & 0x3f
        sport = (h >> 8 ) & 0x3f
        flags = h & 0xff
        
        return cls(src, dst, dport, sport, 
                   prio=prio, flags=flags, endian=endian)

    def to_bytes(self) -> bytes:
        '''
        Serialize CSP header (V1) into a 4-byte sequence
        '''
        # pack upper fields
        value =  (self.prio  & 0x3 ) << 30
        value |= (self.src   & 0x1F) << 25
        value |= (self.dst   & 0x1F) << 20
        value |= (self.dport & 0x3F) << 14
        value |= (self.sport & 0x3F) << 8
        value |=  self.flags
        return value.to_bytes(4, byteorder=self.endian)

    @property
    def hmac(self):
        return bool(self.flags & self.FLAG_HMAC)

    @hmac.setter
    def hmac(self, value):
        if value: self.flags |= self.FLAG_HMAC
        else: self.flags &= ~self.FLAG_HMAC

    @property
    def xtea(self):
        return bool(self.flags & self.FLAG_XTEA)

    @xtea.setter
    def xtea(self, value):
        if value: self.flags |= self.FLAG_XTEA
        else: self.flags &= ~self.FLAG_XTEA

    @property
    def rdp(self):
        return bool(self.flags & self.FLAG_RDP)

    @rdp.setter
    def rdp(self, value):
        if value: self.flags |= self.FLAG_RDP
        else: self.flags &= ~self.FLAG_RDP

    @property
    def crc(self):
        return bool(self.flags & self.FLAG_CRC)

    @crc.setter
    def crc(self, value):
        if value: self.flags |= self.FLAG_CRC
        else: self.flags &= ~self.FLAG_CRC

class CRCEngine:
    '''
    CRC-32C (Castagnoli)
    Big endian is the standard implementation
    '''
    def __init__(self, endian:Literal['big', 'little']='big'):
        self.endian = endian
        self.engine = crc.Calculator(crc.Configuration(
            width=32,
            polynomial=0x1edc6f41,
            init_value=0xffffffff,
            final_xor_value=0xffffffff,
            reverse_input=True,
            reverse_output=True
        ))

    def __call__(self, x: bytes) -> bytes:
        return self.engine.checksum(x).to_bytes(4, self.endian)

class HMACEngine:
    BLOCKSIZE = 64
    DIGESTSIZE = 20

    def __init__(self, key:bytes=b''):
        # Use SHA1 as KDF
        rkey = hashlib.sha1(key).digest()[0:16]
        
        # Normalize key to block size (64 bytes)
        if len(rkey) > self.BLOCKSIZE:
            rkey = hashlib.sha1(rkey).digest()
        rkey = rkey + b'\x00' * (self.BLOCKSIZE - len(rkey))
        
        # Prepare inner and outer padded keys
        self._ipad = bytes(b ^ 0x36 for b in rkey)
        self._opad = bytes(b ^ 0x5C for b in rkey)

    def __call__(self, data:Union[bytes, bytearray, memoryview]):
        sha1 = hashlib.sha1()
        sha1.update(self._ipad)
        sha1.update(data)
        inner_hash = sha1.digest()
        
        outer = hashlib.sha1()
        outer.update(self._opad)
        outer.update(inner_hash)
        return outer.digest()[0:4]

class XTEAEngine:
    def __init__(self, key:bytes=b'', legacy_unsafe:bool=False):
        # Use SHA1 as KDF
        rkey = hashlib.sha1(key).digest()[0:16]
        self.legacy_unsafe = legacy_unsafe

        def counter():
            iv = self.nonce + self.count.to_bytes(4, 'big')
            if self.firstrun:
                self.firstrun = False
            else:
                self.count += 1
                self.count &= 0xffffffff
            return iv

        self.ciper = xtea.new(rkey, mode=xtea.MODE_CTR, counter=counter)

    def encrypt(self, data:Union[bytes, bytearray, memoryview], nonce:bytes[4]):
        assert len(nonce) == 4, 'len(nonce) must be 4'
        self.nonce = nonce
        self.count = 1
        self.firstrun = self.legacy_unsafe
        return self.ciper.encrypt(data)

    def decrypt(self, data:Union[bytes, bytearray, memoryview], nonce:bytes[4]):
        return self.encrypt(data, nonce)

class Packet:
    '''
    CSP packet
    | Field   | Length | Contents       |
    | ------- | ------ | -------------- |
    | Header  | 4B/6B  | V1: 4B, V2: 6B |
    | Payload | N      |                |
    | HMAC    | 4B     | sha1 digest    |
    | CRC     | 4B     | crc checksum   |
    | XTEA    | 4B     | xtea init vec  |
    '''
    
    def __init__(self, 
                 src:int=0, dst:int=0, dport:int=0, sport:int=0,
                 prio:int|Literal['critical', 'high', 'norm', 'low']='norm',
                 payload:bytes=b'',
                 hmac_key:bytes=None, 
                 xtea_key:bytes=None, 
                 rdp:bool=None, 
                 crc:bool=None,
                 flags:int=0,
                 header_endian:Literal['big', 'little']='big',
                 crc_include_header:bool=False,
                 crc_endian:Literal['big', 'little']|None='big',
                 xtea_legacy_unsafe:bool=False,
                 exception:bool=False
                ):
        '''
        set hmac_key=None, xtea_key=None, crc_endian=None for transparent mode
        '''
        self.header = HeaderV1(src, dst, dport, sport, 
                               prio=prio, flags=flags, endian=header_endian,
                               hmac=(hmac_key != None), xtea=(xtea_key != None),
                               rdp=rdp, crc=crc)
        self.hmac_engine = None if hmac_key is None else HMACEngine(hmac_key)
        self.xtea_engine = None if xtea_key is None else XTEAEngine(xtea_key, legacy_unsafe=xtea_legacy_unsafe)
        self.crc_engine = None if crc_endian is None else CRCEngine(endian=crc_endian)
        self.crc_include_header = crc_include_header
        self.exception = exception
        self.payload = payload
    
    def __str__(self):
        xtea_en = self.header.xtea and self.xtea_engine
        hmac_en = self.header.hmac and self.hmac_engine
        crc_en = self.header.crc and self.crc_engine
        size = len(self.payload) + 4 * (xtea_en + hmac_en + crc_en)
        
        return 'Src %d, Dst %d, Dport %d, Sport %d, Pri %d, Flags 0x%d, Size %d' % (
            self.header.src, self.header.dst, self.header.dport, self.header.sport,
            self.header.prio, self.header.flags, size)

    def encode(self) -> bytes:
        header = self.header.to_bytes()
        payload = self.payload
        
        if self.header.hmac and self.hmac_engine:
            payload += self.hmac_engine(payload)
            
        if self.header.crc and self.crc_engine:
            if self.crc_include_header:
                payload += self.crc_engine(header + payload)
            else:
                payload += self.crc_engine(payload)
        
        if self.header.xtea and self.xtea_engine:
            nonce = random.randint(0, 0xffffffff)
            payload = self.xtea_engine.encrypt(self.payload, nonce) + nonce.to_bytes(4, 'big')
        
        return header + payload

    def decode(self, data:bytes):
        self.header = HeaderV1.from_bytes(data[0:4], self.header.endian)
        self.payload = data[4:]
        
        if self.header.xtea and self.xtea_engine:
            nonce = self.payload[-4:]
            if len(nonce) != 4:
                self.payload = b''
                if self.exception: raise ValueError('packet too short')
                return
            
            self.payload = self.xtea_engine.decrypt(self.payload[:-4], nonce)

        if self.header.crc and self.crc_engine:
            crc_val = self.payload[-4:]
            if len(crc_val) != 4:
                self.payload = b''
                if self.exception: raise ValueError('packet too short')
                return
                
            if crc_val != self.crc_engine(self.payload[:-4]):
                self.payload = b''
                if self.exception: 
                    if self.header.xtea and self.xtea_engine:
                        raise ValueError('CRC ERROR after decryption')
                    else:
                        raise ValueError('CRC ERROR')
                return
                    
            self.payload = self.payload[:-4]
        
        if self.header.hmac and self.hmac_engine:
            hmac_val = self.payload[-4:]
            if len(hmac_val) != 4:
                self.payload = b''
                if self.exception: raise ValueError('packet too short')
                return

            if hmac_val != self.hmac_engine(self.payload[:-4]):
                self.payload = b''
                if self.exception: raise ValueError('HMAC ERROR')
                return
                    
            self.payload = self.payload[:-4]
