# Import

import hashlib
import re
import sys
from pathlib import Path

# TODO: Restructure into OOP!!

# Open File
def extract_data_from_file(input_file_path: Path) -> list[str]:
    """ Reads data from binary file, and converts to hex string
    then splits the string every 2 characters

    Args:
        packet_type: The number of packets to extract
        input_file_path: The Path to the input file
    Returns: 
        A list of bytes in hexadecimal

    """
    # regex = re.compile(r"([0-9A-F][0-9A-F] ?)+", re.IGNORECASE)

    packet_to_hex_string: list[str] = []

    with open(input_file_path, "rb") as log_file:
            # for i in range(1):
            line_num = 0
            for line in log_file:
                line_num += 1
                print(f'This is the {line_num} of data')
                hex_string = line.hex()
                print(hex_string)
                # Append hexidecimal string to array of strings
                packet_to_hex_string.append(hex_string)
        
                
            # break

            

    # Return array of hexadecimal strings
    return packet_to_hex_string
               

def search_packet(packet:list[str]) -> list[str]:
    """
    Searches string for the starting seqeunce of an MPI data frame
    OCFFFFOC

    If a data frame is found, the information is added to a data frame array for processing
    
    """
# list[str]:
    data_frames: list[str] = []
    for byte in packet:
        # data_frame_search = re.compile(r"0CFFFF0C([0-9A-F][0-9A-F] ?)+", re.IGNORECASE)
        # match = data_frame_search.search(byte)

        data_frame_search = byte.split(r'0cffff0c')

        # Whenever the sequence 0CFFFF0C is encountered a new list index is created carry the data frame information
        # if match:
        #     data_frame = match.group()

        # 'data_frame_search' is now a LIST of strings
        # To add to a MASS LIST iterate through each index in the 'data_frame_search' array 
        # Append each value to a MASS LIST
        for data_frame in data_frame_search:
            data_frames.append(data_frame)

    return data_frames

def hex_string_to_2n(data_n_frame: str) ->list[str]:
    """
    Takes a string and divides every two characters;
    Allows for indentifying 1 byte of data in the MPI data frame
    """
    data_2n_frame: list[str] = []

    # Split bytes in hexadecimal string
    data_2n_frame = re.findall('..', data_n_frame)

    # print(f'length of data frame now: {len(data_2n_frame)}')

    for byte in data_2n_frame:
        print(len(data_2n_frame))
        print(byte)

    return data_2n_frame

def hex_to_decimal() -> int:
    ''' Converts a hexadecimal string into a decimal'''


def id_bytes_in_data_frame(data_frame: str) ->list[dict]:
    """ Add each data frame to a resulting DICTIONARY???
    """
    # Regex split data into 2s
    mpi_data_frame: list[str] = []
    mpi_data_frame = hex_string_to_2n(data_frame)
    # Convert each string to hex prior to processing (make function)
    mpi_dict_list: list[dict]
    byte_arr: list[str]
    pixel_array: list[int] # Size of the array will be determined by Byte 17 - Byte 16

    print(f'length of MPI DATA FRAME {len(mpi_data_frame)}')
    # Create Dictionary 
    mpi_dictionary = dict()
    # keys = ['Sync Byte 1', 'Sync Byte 2', 'Sync Byte 3', 'Sync Byte 4',
    #         'Frame Counter', 'Board Temperature', 'Firmware Version', 
    #         'Detector Status', 'Inner Dome Voltage Setting', 'Inner Dome Scan Index',
    #         'Inner Dome Voltage ADC Reading', 'First Pixel Index', 'Last Pixel Index', 
    #         'Integration Period'
    #         ]

    # Byte 0 - Sync Byte
    mpi_dictionary.update({"Sync Byte 1": mpi_data_frame[0]})
    # Byte 1 - Sync Byte
    mpi_dictionary.update({"Sync Byte 2": mpi_data_frame[1]})
    # Byte 2 - Sync Byte
    mpi_dictionary.update({"Sync Byte 3": mpi_data_frame[2]})
    # Byte 3 - Sync Byte
    mpi_dictionary.update({"Sync Byte 4": mpi_data_frame[3]})

    # Byte 4 & 5 - Frame Counter
    byte_4 = int(mpi_data_frame[4], 16) # Convert byte 4 hexadecimal string to decimal type 'int'
    byte_5 = int(mpi_data_frame[5], 16) # Convert byte 5 hexadecimal string to decimal type 'int'
    counter = (byte_4*256) + byte_5

    # Update MPI Data Frame Dictionary with Frame Number Element
    mpi_dictionary.update({"Frame Number": counter})

    # Byte 6  & 7 - Board Temperature
    byte_6 = int(mpi_data_frame[6], 16)
    byte_7 = int(mpi_data_frame[7], 16)

    temperature = ((byte_6)*256 + byte_7) / 128.0

    # Update MPI Data Frame Dictionary with Board Temperature Element
    mpi_dictionary.update({"Board Temperature": temperature})

    # Byte 8 - Firmware version

    # Byte 9
    # Byte 10
    # Byte 10
    # Byte 11
    # Byte 12
    # Byte 13
    # Byte 14
    # Byte 15
    # Byte 16 (Defines FIRST pixel index)
    # Byte 17 (Defines LAST pixel index)
    # Byte 18
    # Byte 19
    # Byte 20 & 21
    # Byte 22 & 23
    # ...
    # CRC Bytes

    mpi_dict_list.append(mpi_dictionary)

    return mpi_dict_list
        

# def reconstruct_bulk_downlinked_file(input_log_file_path: Path, output_file_path: Path) -> None:
#     """Reconstructs a bulk downlinked file from the log file.

#     Args:
#         log_file_path: Path to the log file.
#         output_file_path: Path to save the reconstructed file.
#     """
#     byte_offset_list: list[int] = []
#     with open(output_file_path, "w") as output_file:
#         for packet in extract_data_from_file(0x10, input_log_file_path):
#             # Read the offset in the file from bytes 5,6,7,8.
#             # offset = int.from_bytes(packet[5:9], "little")
#             # byte_offset_list.append(offset)

#             # Write the packet to the output file.
#             output_file.write(packet)

#             # if len(byte_offset_list) == 1:
#             #     print(f"First packet data: offset_bytes={offset}, length_bytes={len(packet[9:])}")

#     # Check if all packets are present
#     if len(byte_offset_list) != len(set(byte_offset_list)):
#         print("Warning: Duplicate packet byte offsets found.")

#     if byte_offset_list != sorted(byte_offset_list):
#         print("Warning: Packet byte offsets are not in order.")


def main() -> None:
    print("Program Start")
    if len(sys.argv) != 2:
        print(
            "Usage: python extract_radio_packets_from_logs.py <log_file_path> <output_file_path>"
        )
        sys.exit(1)

    log_file_path = Path(sys.argv[1])

    if not log_file_path.exists():
        print(f"Log file does not exist: {log_file_path}")
        sys.exit(1)

    # Open the parsed MPI binary data file and read into an array of strings
    packets_in_hex_string = extract_data_from_file(sys.argv[1])
    print('We have now returned to main')

    # FOR DEBUGGING: Print out the array of strings
    # data_num = 0
    # for data in packets_in_hex_string:
    #     # data_num +=1
    #     print(f'This is the {data_num} hexadecimal character')
    #     print(len(packets_in_hex_string))
    #     print(data)
    # END of DEBUGGING

    # Search each line of the packet for MPI Data Frames
    frame = search_packet(packets_in_hex_string)

    # FOR DEBUGGING: Print out the Data Frames
    # data_num = 0
    # for frame_stuff in frame:
    #     data_num +=1
    #     print(f'This is the {data_num} frame')
    #     print(len(frame))
    #     print(frame_stuff)
    # END of DEBUGGING

    # Parse through each index of FRAME and decode each byte to human readable format
    # for frame_index in frame:

    id_bytes_in_data_frame(frame[1])


    
if __name__ == "__main__":
    main()




# Read Binary File
# Convert Binary Values to Hexadecimal and save to variable
# Print Data


##### CUTS #############
### From 'extract data from file'
 # Search for start of data frame
# Note that there may be more than one data frame in the same line
# ALL Data Frames need to be accounted for

# match = regex.search(line.hex())

# Convert the line to hexadecimal string
# print('This is the line in HEX')
# print(hex_string)

# data_frame_search = re.compile(r"0CFFFF0C([0-9A-F][0-9A-F] ?)+", re.IGNORECASE)
# match = data_frame_search.search(hex_string)

# if match:
#     data_frame = match.group()

# Convert line to a hex string then group by 2 before appending
# packet_to_hex_string = re.findall('..', hex_string)
###
 # if match:
                #     hex_string = match.group().replace("", " ")
                #     # hex_string = line.hex()
                #     # packet_bytes = bytes.fromhex(hex_string)
                #     hex_string = line.hex()
                #     print(hex_string)
                #     # packets.append(packet_bytes)
                #     packets.append(hex_string)
                #     break