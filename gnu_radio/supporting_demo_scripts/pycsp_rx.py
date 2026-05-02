# In[1]:

import struct
from pathlib import Path

import pycsp as csp
import pycsplink as csplink
from loguru import logger

# In[ ]:


OBC_ADDR = 1
EPS_ADDR = 2
TTC_ADDR = 5
CAM_ADDR = 6
TNC_ADDR = 9
GCS_ADDR = 10


# In[ ]:


DPORT_CMP = 0
DPORT_PING = 1
DPORT_PS = 2
DPORT_MEMFREE = 3
DPORT_REBOOT = 4
DPORT_BUF_FREE = 5
DPORT_UPTIME = 6


# In[ ]:


def parse_obc_downlink(data: bytes):
    if data[0] == 3:
        return data[1:].decode()

    if data[0] == 4:
        if len(data) < 13:
            raise ValueError("packet too short")

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

    return data


# In[ ]:


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


# In[ ]:


if ttc is not None:
    ttc.close()
ttc = csplink.GrcLink()


# In[ ]:


while True:
    ttc = csplink.GrcLink(timeout=1)

    try:
        rx = ttc.recv()
        # filter echo packets
        # TODO: fix this dirty impl
        if csp.HeaderV1.from_bytes(rx[0:4]).src == GCS_ADDR:
            continue

        # decode packets
        resp = downlink.decode(rx)
        if not resp:
            logger.info(resp)
            continue

        if resp.header.src == OBC_ADDR and resp.header.dst == GCS_ADDR:
            logger.info(parse_obc_downlink(resp.payload))
        else:
            logger.info(resp, resp.payload.hex())

    except ValueError as e:
        logger.error(e)
    except TimeoutError:
        pass
    except KeyboardInterrupt:
        pass

    ttc.close()


# In[ ]:
