import time
from pathlib import Path
from typing import Literal

import pycsp as csp
import pycsplink as csplink
from loguru import logger

# In[7]:


OBC_ADDR = 1
EPS_ADDR = 2
TTC_ADDR = 5
CAM_ADDR = 6
TNC_ADDR = 9
GCS_ADDR = 10


# In[8]:


DPORT_CMP = 0
DPORT_PING = 1
DPORT_PS = 2
DPORT_MEMFREE = 3
DPORT_REBOOT = 4
DPORT_BUF_FREE = 5
DPORT_UPTIME = 6


# In[11]:

hmac_key_file_path = Path("hmac_key.txt")
if hmac_key_file_path.exists():
    hmac_key = bytes.fromhex(hmac_key_file_path.read_text().strip())
else:
    logger.warning("WARNING: Using fake HMAC key as hmac_key.txt does not exist.")
    hmac_key = bytes.fromhex("ABCDABCDABCDABCDABCDABCDABCDABCD")

uplink = csplink.AX100(
    hmac_key=hmac_key,
    crc=False,
    reed_solomon=True,
    randomize=True,
    len_field=True,
    syncword=True,
    prefill=32,
    tailfill=1,
)
downlink = csplink.AX100(
    hmac_key=None,
    crc=True,
    reed_solomon=False,
    randomize=False,
    len_field=False,
    syncword=False,
    exception=False,
    verbose=True,
)

ttc = None


# In[12]:


if ttc is not None:
    ttc.close()
ttc = csplink.GrcLink(timeout=1)


# In[13]:


def cts_ping(dst: int = TTC_ADDR):
    SPORT = 16  # 0..63
    packet = csp.Packet(
        GCS_ADDR, dst, DPORT_PING, SPORT, prio="norm", hmac_key=None, crc=False
    )
    packet.payload = bytes.fromhex("00010203040506070809")

    # ttc.send(uplink.encode(packet), b'')
    ttc.send(uplink.encode(packet), b"")
    """try:
        ttc.recv() # receive echo
        rx = ttc.recv(1)
        resp = downlink.decode(rx)
        print(resp, resp.payload.hex() if resp else None)
    except TimeoutError:
        print('TIMEOUT')"""


def cts_send(cmd: str, dst: int = OBC_ADDR):
    print(f'cts_send("{cmd})"')

    SPORT = 16  # 0..63
    DPORT = 7
    packet = csp.Packet(
        GCS_ADDR, dst, DPORT, SPORT, prio="norm", hmac_key=None, crc=False
    )
    packet.payload = cmd.encode("ascii")

    ttc.send(uplink.encode(packet), b"")
    # ttc.recv(1) # receive echo


def cts_query(prop: Literal["process", "memfree", "buffree", "uptime"], dst=TTC_ADDR):
    dport = {
        "process": DPORT_PS,
        "memfree": DPORT_MEMFREE,
        "buffree": DPORT_BUF_FREE,
        "uptime": DPORT_UPTIME,
    }[prop]

    SPORT = 16  # 0..63
    packet = csp.Packet(
        GCS_ADDR, dst, dport, SPORT, prio="norm", hmac_key=None, crc=True
    )

    ttc.send(uplink.encode(packet), b"")
    ttc.recv(1)  # receive echo
    try:
        rx = ttc.recv(1)
        resp = downlink.decode(rx)
        val = int.from_bytes(resp.payload, "big")
    except TimeoutError:
        val = None

    return val


# In[14]:


# def ax100_param_dump(addr='all', ax100_addr=TTC_ADDR):
#     AX100_PORT_RPARAM = 7
#     PARAM_PULL_ALL_REQUEST = 4

#     include_mask = 0xffffffff
#     exclude_mask = 0

#     SPORT = 16 # 0..63
#     packet = csp.Packet(GCS_ADDR, ax100_addr, AX100_PORT_RPARAM, SPORT, crc=True)
#     packet.payload = bytes([PARAM_PULL_ALL_REQUEST, 0, 0, 0])
#     packet.payload += include_mask.to_bytes(4, 'big')
#     packet.payload += exclude_mask.to_bytes(4, 'big')

#     ttc.send(uplink.encode(packet), b'')
#     ttc.recv(1) # receive echo
#     try:
#         rx = ttc.recv(1)
#         resp = downlink.decode(rx)
#         val = int.from_bytes(resp.payload, 'big')
#     except TimeoutError:
#         val = None
#         pass
#
# DOES NOT WORK


uplink = csplink.AX100(
    hmac_key=hmac_key,
    crc=False,
    reed_solomon=True,
    randomize=True,
    len_field=True,
    syncword=True,
    prefill=32,
    tailfill=1,
)
downlink = csplink.AX100(
    hmac_key=None,
    crc=True,
    reed_solomon=False,
    randomize=False,
    len_field=False,
    syncword=False,
    exception=False,
    verbose=True,
)

ttc = None


# In[16]:


if ttc is not None:
    ttc.close()
ttc = csplink.GrcLink()


# In[ ]:


for _ in range(10):
    cts_send("CTS1+hello_world()!")
    time.sleep(0.2)


# In[ ]:


if 0:
    [print(key, cts_query(key)) for key in ["process", "memfree", "buffree", "uptime"]]


# In[ ]:


cts_send("CTS1+hello_world()!")


# In[ ]:


cts_send("CTS1+fs_mount()!")


# In[ ]:


cts_send("CTS1+fs_list_directory(/,0,10)!")


# In[ ]:


# ax100_param_dump('all')


# In[ ]:
while True:
    cts_send(input())
