# Import

import hashlib
import re
import sys
from pathlib import Path

# TODO: Restructure into OOP!!

# TODO:
"""
Requirements:
- Must read parsed MPI downlinked data
- Must process response code to a telecommand, XX (254_10 or 0xfe_16) _ = BASE #
- Must parse through each MPI Data Frame by READING and RETAINING the start tag 0c ff ff 0c (DISREGARD FOR CASE)
- Must input all pixel data into an array for analysis
- Must plot pixel data 


"""

# Open File
def extract_data_from_file(input_file_path: Path) -> str:
    """ Reads data from binary file, and converts to hex string
    then splits the string every 2 characters

    Args:
        packet_type: The number of packets to extract
        input_file_path: The Path to the input file
    Returns: 
        A list of bytes in hexadecimal

    """
    # regex = re.compile(r"([0-9A-F][0-9A-F] ?)+", re.IGNORECASE)

    # 
    packet_to_hex_string: list[str] = []

    # Open parsed source file for extraction of MPI Data
    with open(input_file_path, "rb") as log_file:
            # for i in range(1):

            # DEBUGGING LINE
            # line_num = 0
            # DEBUGGING LINE END

            # Iterate through each line in the log file
            for line in log_file:
                
                """
                # DEBUGGING ONLY:
                line_num += 1
                print(f'This is the {line_num} of data')
                
                """
                # Convert the binary data in each line to HEX STRING
                hex_string = line.hex()

                """
                # DEBUGGING ONLY
                print(hex_string)
                Append hexidecimal string to array of strings
                packet_to_hex_string.join(hex_string)
                """

                # 
                packet_to_hex_string.append(hex_string)
                 
            single_packet_to_string = ''.join(packet_to_hex_string)
            
 
            # break

    # Return array of hexadecimal strings
    return single_packet_to_string
               

def search_packet(packet:str) -> list[str]:
    """
    Searches string for the starting seqeunce of an MPI data frame
    OCFFFFOC

    If a data frame is found, the information is added to a data frame array for processing
    
    """
# list[str]:
    data_frames: list[str] = []
    # for byte in packet:
        # data_frame_search = re.compile(r"0CFFFF0C([0-9A-F][0-9A-F] ?)+", re.IGNORECASE)
        # match = data_frame_search.search(byte)

    data_frame_search = packet.split(r'0cffff0c')


    # print(data_frame_search)

        # Whenever the sequence 0CFFFF0C is encountered a new list index is created carry the data frame information
        # if match:
        #     data_frame = match.group()

        # 'data_frame_search' is now a LIST of strings
        # To add to a MASS LIST iterate through each index in the 'data_frame_search' array 
        # Append each value to a MASS LIST
    # for data_frame in data_frame_search:
    #     data_frames.append(data_frame)
            
        # print(data_frames)
    return data_frame_search

def hex_string_to_2n(data_n_frame: str) ->list[str]:
    """
    Takes a string and divides every two characters;
    Allows for indentifying 1 byte of data in the MPI data frame

    """
    data_2n_frame: list[str] = []

    # Split bytes in hexadecimal string
    data_2n_frame = re.findall('..', data_n_frame)

    # print(f'length of data frame now: {len(data_2n_frame)}')

    # for byte in data_2n_frame:
    #     print(len(data_2n_frame))
    #     print(byte)

    # print(data_2n_frame)

    return data_2n_frame

def hex_to_decimal() -> int:
    ''' Converts a hexadecimal string into a decimal'''

def pixel_value(pixel_data_1: int, pixel_data_2: int) -> int:
    """  
    Calculate the value of the pixel

    Args: 
        pixel_data_1: First byte defined for pixel
        pixel_data_2: Second byte defined for pixel
    Returns:
        pixel_val: Value of the pixel for the 
    """
    pixel_val = pixel_data_1 + pixel_data_2

    return pixel_val


def id_bytes_in_data_frame(data_frame: list[str]) ->list[dict]:
    """ Add each data frame to a resulting DICTIONARY???
    """
    #TODO: Test if data does calculations directly in hex or needs to be converted into decimal
    # Regex split data into 2s
    


    mpi_data_frame: list[str] = []
    index = 0 
    for hex_val in data_frame:
        mpi_data_frame.append(hex_val) 
       
        
    
    mpi_data_frame = hex_string_to_2n(data_frame)
    # print(mpi_data_frame)
    # Convert each string to hex prior to processing (make function)
    mpi_dict_list: list[dict] = []
    byte_arr: list[str]
    pixel_array: list[int] # Size of the array will be determined by Byte 17 - Byte 16

    # print(f'length of MPI DATA FRAME {len(mpi_data_frame)}')
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
    byte_4 = int(mpi_data_frame[0], 16) # Convert byte 4 hexadecimal string to decimal type 'int'
    byte_5 = int(mpi_data_frame[1], 16) # Convert byte 5 hexadecimal string to decimal type 'int'
    counter = (byte_4*256) + byte_5
    print(byte_4)
    print(byte_5)

    # Update MPI Data Frame Dictionary with Frame Number Element
    mpi_dictionary.update({"Frame Number": counter})

    # Byte 6  & 7 - Board Temperature
    byte_6 = int(mpi_data_frame[2], 16)
    byte_7 = int(mpi_data_frame[3], 16)

    temperature = ((byte_6)*256 + byte_7) / 128.0

    # Update MPI Data Frame Dictionary with Board Temperature Element
    mpi_dictionary.update({"Board Temperature": temperature})

    # Byte 8 - Firmware version
    byte_8 = int(mpi_data_frame[4], 16)
    mpi_dictionary.update({"Firmware Version": byte_8})

    # Byte 9 & 10 - Detector Status
    byte_9 = int(mpi_data_frame[5], 16)
    byte_10 = int(mpi_data_frame[6], 16)

    detector_status = byte_9*256 + byte_10

    mpi_dictionary.update({"Detector Status": detector_status})
    
    
    # Byte 11 & 12 - Inner Dome Voltage Setting
    byte_11 = int(mpi_data_frame[7], 16)
    byte_12 = int(mpi_data_frame[8], 16)

    # Set inner dome scan
    inner_dome_v_setting = byte_11 * 256 + byte_12

    mpi_dictionary.update({"Inner Dome Voltage Setting": inner_dome_v_setting})
    

    # Byte 13 - Inner Dome Scan Index
    byte_13 = int(mpi_data_frame[9], 16)

    inner_dome_scan_index = byte_13
    mpi_dictionary.update({"Inner Dome Scan Index": inner_dome_scan_index})

    # Byte 14 & 15 - Inner Dome Voltage ADC Reading
    byte_14 = int(mpi_data_frame[10], 16)
    byte_15 = int(mpi_data_frame[11], 16)

    
    eu = byte_14*256 + byte_15
    # v_id = (float)(eu & ('0x3ff'))*0.105361 - 101.808

    # mpi_dictionary.update({"Inner Dome Voltage ADC Reading": v_id})

    # Byte 16 (Defines FIRST pixel index)
    byte_16 = int(mpi_data_frame[12], 16)
    # Byte 17 (Defines LAST pixel index)
    byte_17 = int(mpi_data_frame[13], 16)
    
    mpi_dictionary.update({"First Pixel Index": byte_16})
    mpi_dictionary.update({"Last Pixel Index": byte_17})

    # Determine Length of Pixel Data
    pixel_len = byte_17 - byte_16

    # Byte 18 & 19 - Integration Period

    byte_18 = int(mpi_data_frame[14], 16)
    byte_19 = int(mpi_data_frame[15], 16)

    integration_period_set = byte_18*256 + byte_19

    mpi_dictionary.update({"Integration Period": integration_period_set})


    # Byte 20 & 21 - First Pixel Index

    byte_20 = int(mpi_data_frame[16], 16)
    byte_21 = int(mpi_data_frame[17], 16)

    #TODO: Create a function that determines the value of the pixel
    # for loop
    # update pixel array with values
   

    first_pixel = byte_20 + byte_21


    # Byte 22 & 23 - Second Pixel Index
    # ...
    # CRC Bytes
    crc_byte_1 = 0 # mpi_data_frame[pixel_len - 2]
    crc_byte_2 = 0 # mpi_data_frame[pixel_len - 1]

    crc_check = crc_byte_1 + crc_byte_2
    
    mpi_dictionary.update({"CRC Check": crc_check})

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
    listy: list[dict] = []
    index = 1
    for data in frame:
        print(f'Data Length {len(data)}')
        print(data)
        if (len(data) <= 8):
            continue
        listy.append(id_bytes_in_data_frame(data))
        index += 1

    print(len(listy))
    # print(listy)
    

    j = 0
    while j < 150: #len(listy)
        print()
        for key, value in listy[j][0].items():
            print(f"{key}: {value}")
            j += 1
            # print(j)


    
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