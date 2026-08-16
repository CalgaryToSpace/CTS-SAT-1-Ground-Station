# MPI Telecommand Codes

tc_code = {
"""
Telecommand Codes for the MPI
Note that all KEYS are in hexadecimal
"0x" has been ommitted from all entries, but assume that all KEYS translate as follows:

    Documentation says that "0x13" = ENABLE_SCIENCE_TELEMETRY
    In this dictionary, this will be represented as "13"

"""
    "01": "SET_INNER_DOME_SCAN_MODE",
    "02": "SET_SCAN_MIDOINT_VOLTAGE",
    "03": "SET_SCAN_AMPLITUDE_VOLTAGE",
    "04": "SET_SCAN_NUMBER_OF_STEPS",
    "05": "SET_INTEGRATION_PERIOD",
    "06": "AGC_ENABLE",
    "07": "AGC_DISABLE",
    "08": "BL_ENTER_BOOTLOADER",
    "09": "BL_EXIT_BOOTLOADER",
    "0A": "BL_SET_APPSW_CRC",
    "0B": "BL_ERASE_TEMP_APP_MEMORY",
    "0C": "BL_STORE_APPSW_CHUNK",
    "0D": "BL_UPGRADE_FIRMWARE",
    "0E": "AGC_SET_THRESHOLDS",
    "0F": "AGC_SET_PIXEL_RANGE",
    "10": "AGC_SET_NUMBER_OF_MAXIMA_FOR_CONTROL_VALUE",
    "11": "AGC_SET_NUMBER_OF_PIXELS_FOR_BASELINE_VALUE",
    "12": "SET_BASELINE_FIRST_PIXEL",
    "13": "ENABLE_SCIENCE_TELEMETRY",
    "14": "DISABLE_SCIENCE_TELEMETRY",
    "15": "TM_SET_PIXEL_RANGE",
    "16": "SET_MIN_INTEGRATION_PERIOD",
    "17": "SET_MAX_INTEGRATION_PERIOD",
    "18": "SET_AGC_INTEGRATION_STEP",
    "255": "Invalid Telecommand"

}