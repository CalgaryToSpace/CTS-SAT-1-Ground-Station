#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pycsp as csp
import pycsplink as csplink
import socket
import time

from typing import Literal
from types import SimpleNamespace
import struct


# In[ ]:


OBC_ADDR=1
EPS_ADDR=2
TTC_ADDR=5
CAM_ADDR=6
TNC_ADDR=9
GCS_ADDR=10


# In[ ]:


DPORT_CMP = 0
DPORT_PING = 1
DPORT_PS = 2
DPORT_MEMFREE = 3
DPORT_REBOOT = 4
DPORT_BUF_FREE = 5
DPORT_UPTIME = 6


# In[ ]:


class GrcLink:
    def __init__(self, addr='127.0.0.1', port=52001, mtu=1024, timeout=1):
        self.s = socket.create_connection((addr, port))
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


# In[ ]:


def parse_obc_downlink(data):
    if data[0] == 3:
        return data[1:].decode()

    elif data[0] == 4:
        if len(data) < 13:
            raise ValueError('packet too short')

        # Unpack header (big-endian; change '>' to '<' for little-endian)
        tssent, response_code, duration_ms, seq_num, total_packets = struct.unpack(
            ">Q B H B B", data[:13]
        )

        # Extract and decode content
        raw_content = data[13:200]
        content = raw_content.split(b"\x00", 1)[0].decode("ascii", errors="replace")

        return {
            "tssent": tssent,
            "response_code": response_code,
            "duration_ms": duration_ms,
            "sequence_number": seq_num,
            "total_packets": total_packets,
            "content": content,
        }

    else:
        return data


# In[ ]:


with open('hmac_key.txt', 'r') as f:
    hmac_key = bytes.fromhex(f.read().strip())

uplink = csplink.AX100(hmac_key=hmac_key, crc=False, reed_solomon=True, randomize=True, 
                       len_field=True, syncword=True, prefill=32, tailfill=1)
downlink = csplink.AX100(hmac_key=None, crc=True, reed_solomon=False, randomize=False, 
                       len_field=False, syncword=False, exception=False, verbose=True)

ttc = None


# In[ ]:


if not ttc is None: ttc.close()
ttc = GrcLink()


# In[ ]:


#ttc.close()


# In[ ]:


while True:
    ttc = GrcLink(timeout=1)

    try:
        rx = ttc.recv()
        # filter echo packets
        # TODO: fix this dirty impl
        if csp.HeaderV1.from_bytes(rx[0:4]).src == GCS_ADDR:
            continue

        # decode packets
        resp = downlink.decode(rx)
        if not resp:
            print(resp)
            continue

        if resp.header.src == OBC_ADDR and resp.header.dst == GCS_ADDR:
            print(parse_obc_downlink(resp.payload))
        else:
            print(resp, resp.payload.hex())

    except ValueError as e:
        print(e)
    except TimeoutError:
        pass
    except KeyboardInterrupt:
        pass

    ttc.close()


# In[ ]:




