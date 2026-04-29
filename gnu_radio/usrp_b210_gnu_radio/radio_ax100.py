#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: AX100 Radio Link
# Author: Shu Luo
# GNU Radio version: 3.10.6.0

from packaging.version import Version as StrictVersion
from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import analog
from gnuradio import blocks
from gnuradio import digital
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import gr, pdu
from gnuradio import network
from gnuradio import uhd
import time
from gnuradio.qtgui import Range, RangeWidget
from PyQt5 import QtCore
import math
import satellites.components.deframers
import satellites.components.demodulators
import sip



class radio_ax100(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "AX100 Radio Link", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("AX100 Radio Link")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "radio_ax100")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.baud_rate = baud_rate = 9600
        self.freq_uncertainty = freq_uncertainty = 20e3
        self.fdev = fdev = baud_rate/4
        self.samp_rate = samp_rate = 115200*4
        self.bw = bw = freq_uncertainty+2*fdev+baud_rate
        self.ratio = ratio = 2**int(math.log2(samp_rate/max(bw, baud_rate)))
        self.iq_rate = iq_rate = samp_rate/ratio
        self.duc_taps = duc_taps = firdes.low_pass(1.0, samp_rate, iq_rate/2,iq_rate/2*0.2, window.WIN_HAMMING, 6.76)
        self.ddc_taps = ddc_taps = firdes.low_pass(1.0, samp_rate, bw/2,bw/2*0.2, window.WIN_HAMMING, 6.76)
        self.tx_pwr_cal = tx_pwr_cal = 65.5
        self.tx_pwr = tx_pwr = 0
        self.sps = sps = iq_rate/baud_rate
        self.rx_gain = rx_gain = 65
        self.freq = freq = 436.15e6
        self.duc_actual_taps = duc_actual_taps = len(duc_taps)/ratio
        self.ddc_actual_taps = ddc_actual_taps = len(ddc_taps)/ratio

        ##################################################
        # Blocks
        ##################################################

        self._tx_pwr_range = Range(-10, 13, 0.1, 0, 200)
        self._tx_pwr_win = RangeWidget(self._tx_pwr_range, self.set_tx_pwr, "Tx Power", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._tx_pwr_win, 2, 0, 1, 1)
        for r in range(2, 3):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._rx_gain_range = Range(0, 76, 1, 65, 200)
        self._rx_gain_win = RangeWidget(self._rx_gain_range, self.set_rx_gain, "Rx Gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._rx_gain_win, 1, 0, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._freq_range = Range(430e6, 440e6, 1e2, 436.15e6, 200)
        self._freq_win = RangeWidget(self._freq_range, self.set_freq, "Freq", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._freq_win, 0, 0, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(("", '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
        )
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_time_unknown_pps(uhd.time_spec(0))

        self.uhd_usrp_source_0.set_center_freq(freq-samp_rate/4, 0)
        self.uhd_usrp_source_0.set_antenna("RX2", 0)
        self.uhd_usrp_source_0.set_gain(rx_gain, 0)
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join(("", '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
            "",
        )
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        self.uhd_usrp_sink_0.set_time_unknown_pps(uhd.time_spec(0))

        self.uhd_usrp_sink_0.set_center_freq(freq, 0)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0.set_gain(tx_pwr+tx_pwr_cal, 0)
        self.satellites_fsk_demodulator_0 = satellites.components.demodulators.fsk_demodulator(baudrate = baud_rate, samp_rate = iq_rate, iq = True, subaudio = False, options="")
        self.satellites_ax100_deframer_0 = satellites.components.deframers.ax100_deframer(mode = "ASM", scrambler = "CCSDS", syncword_threshold = 1, options="")
        self.qtgui_time_sink_x_0_0 = qtgui.time_sink_f(
            (baud_rate//10), #size
            baud_rate, #samp_rate
            "", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0_0.set_y_axis(-1.2, 1.2)

        self.qtgui_time_sink_x_0_0.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_0_0.enable_tags(True)
        self.qtgui_time_sink_x_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_0_0.enable_autoscale(False)
        self.qtgui_time_sink_x_0_0.enable_grid(True)
        self.qtgui_time_sink_x_0_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0_0.enable_stem_plot(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_time_sink_x_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_sink_x_0_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_time_sink_x_0_0_win, 4, 0, 1, 1)
        for r in range(4, 5):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            1024, #size
            window.WIN_FLATTOP, #wintype
            (freq-samp_rate/4), #fc
            samp_rate, #bw
            "", #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis((-120), (-40))
        self.qtgui_freq_sink_x_0.set_y_label('Amplitude', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(True)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(True)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_freq_sink_x_0_win, 3, 0, 1, 1)
        for r in range(3, 4):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.pdu_pdu_to_tagged_stream_0 = pdu.pdu_to_tagged_stream(gr.types.byte_t, 'packet_len')
        self.network_socket_pdu_0 = network.socket_pdu('TCP_SERVER', '', '52001', 10000, False)
        self.interp_fir_filter_xxx_0 = filter.interp_fir_filter_ccc(ratio, duc_taps)
        self.interp_fir_filter_xxx_0.declare_sample_delay(0)
        self.freq_xlating_fir_filter_xxx_0 = filter.freq_xlating_fir_filter_ccc(ratio, ddc_taps, (samp_rate/4), samp_rate)
        self.digital_chunks_to_symbols_xx_0 = digital.chunks_to_symbols_bf([-1, 1], 1)
        self.blocks_unpack_k_bits_bb_0 = blocks.unpack_k_bits_bb(8)
        self.blocks_repeat_0 = blocks.repeat(gr.sizeof_char*1, int(sps))
        self.blocks_message_debug_0 = blocks.message_debug(True)
        self.analog_frequency_modulator_fc_0 = analog.frequency_modulator_fc((2*math.pi*fdev/iq_rate))


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.network_socket_pdu_0, 'pdus'), (self.pdu_pdu_to_tagged_stream_0, 'pdus'))
        self.msg_connect((self.satellites_ax100_deframer_0, 'out'), (self.blocks_message_debug_0, 'print'))
        self.msg_connect((self.satellites_ax100_deframer_0, 'out'), (self.network_socket_pdu_0, 'pdus'))
        self.connect((self.analog_frequency_modulator_fc_0, 0), (self.interp_fir_filter_xxx_0, 0))
        self.connect((self.blocks_repeat_0, 0), (self.digital_chunks_to_symbols_xx_0, 0))
        self.connect((self.blocks_unpack_k_bits_bb_0, 0), (self.blocks_repeat_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0, 0), (self.analog_frequency_modulator_fc_0, 0))
        self.connect((self.freq_xlating_fir_filter_xxx_0, 0), (self.satellites_fsk_demodulator_0, 0))
        self.connect((self.interp_fir_filter_xxx_0, 0), (self.uhd_usrp_sink_0, 0))
        self.connect((self.pdu_pdu_to_tagged_stream_0, 0), (self.blocks_unpack_k_bits_bb_0, 0))
        self.connect((self.satellites_fsk_demodulator_0, 0), (self.qtgui_time_sink_x_0_0, 0))
        self.connect((self.satellites_fsk_demodulator_0, 0), (self.satellites_ax100_deframer_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.freq_xlating_fir_filter_xxx_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.qtgui_freq_sink_x_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "radio_ax100")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_baud_rate(self):
        return self.baud_rate

    def set_baud_rate(self, baud_rate):
        self.baud_rate = baud_rate
        self.set_bw(self.freq_uncertainty+2*self.fdev+self.baud_rate)
        self.set_fdev(self.baud_rate/4)
        self.set_ratio(2**int(math.log2(self.samp_rate/max(self.bw, self.baud_rate))))
        self.set_sps(self.iq_rate/self.baud_rate)
        self.qtgui_time_sink_x_0_0.set_samp_rate(self.baud_rate)

    def get_freq_uncertainty(self):
        return self.freq_uncertainty

    def set_freq_uncertainty(self, freq_uncertainty):
        self.freq_uncertainty = freq_uncertainty
        self.set_bw(self.freq_uncertainty+2*self.fdev+self.baud_rate)

    def get_fdev(self):
        return self.fdev

    def set_fdev(self, fdev):
        self.fdev = fdev
        self.set_bw(self.freq_uncertainty+2*self.fdev+self.baud_rate)
        self.analog_frequency_modulator_fc_0.set_sensitivity((2*math.pi*self.fdev/self.iq_rate))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_ddc_taps(firdes.low_pass(1.0, self.samp_rate, self.bw/2, self.bw/2*0.2, window.WIN_HAMMING, 6.76))
        self.set_duc_taps(firdes.low_pass(1.0, self.samp_rate, self.iq_rate/2, self.iq_rate/2*0.2, window.WIN_HAMMING, 6.76))
        self.set_iq_rate(self.samp_rate/self.ratio)
        self.set_ratio(2**int(math.log2(self.samp_rate/max(self.bw, self.baud_rate))))
        self.freq_xlating_fir_filter_xxx_0.set_center_freq((self.samp_rate/4))
        self.qtgui_freq_sink_x_0.set_frequency_range((self.freq-self.samp_rate/4), self.samp_rate)
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_center_freq(self.freq-self.samp_rate/4, 0)

    def get_bw(self):
        return self.bw

    def set_bw(self, bw):
        self.bw = bw
        self.set_ddc_taps(firdes.low_pass(1.0, self.samp_rate, self.bw/2, self.bw/2*0.2, window.WIN_HAMMING, 6.76))
        self.set_ratio(2**int(math.log2(self.samp_rate/max(self.bw, self.baud_rate))))

    def get_ratio(self):
        return self.ratio

    def set_ratio(self, ratio):
        self.ratio = ratio
        self.set_ddc_actual_taps(len(self.ddc_taps)/self.ratio)
        self.set_duc_actual_taps(len(self.duc_taps)/self.ratio)
        self.set_iq_rate(self.samp_rate/self.ratio)

    def get_iq_rate(self):
        return self.iq_rate

    def set_iq_rate(self, iq_rate):
        self.iq_rate = iq_rate
        self.set_duc_taps(firdes.low_pass(1.0, self.samp_rate, self.iq_rate/2, self.iq_rate/2*0.2, window.WIN_HAMMING, 6.76))
        self.set_sps(self.iq_rate/self.baud_rate)
        self.analog_frequency_modulator_fc_0.set_sensitivity((2*math.pi*self.fdev/self.iq_rate))

    def get_duc_taps(self):
        return self.duc_taps

    def set_duc_taps(self, duc_taps):
        self.duc_taps = duc_taps
        self.set_duc_actual_taps(len(self.duc_taps)/self.ratio)
        self.interp_fir_filter_xxx_0.set_taps(self.duc_taps)

    def get_ddc_taps(self):
        return self.ddc_taps

    def set_ddc_taps(self, ddc_taps):
        self.ddc_taps = ddc_taps
        self.set_ddc_actual_taps(len(self.ddc_taps)/self.ratio)
        self.freq_xlating_fir_filter_xxx_0.set_taps(self.ddc_taps)

    def get_tx_pwr_cal(self):
        return self.tx_pwr_cal

    def set_tx_pwr_cal(self, tx_pwr_cal):
        self.tx_pwr_cal = tx_pwr_cal
        self.uhd_usrp_sink_0.set_gain(self.tx_pwr+self.tx_pwr_cal, 0)

    def get_tx_pwr(self):
        return self.tx_pwr

    def set_tx_pwr(self, tx_pwr):
        self.tx_pwr = tx_pwr
        self.uhd_usrp_sink_0.set_gain(self.tx_pwr+self.tx_pwr_cal, 0)

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.blocks_repeat_0.set_interpolation(int(self.sps))

    def get_rx_gain(self):
        return self.rx_gain

    def set_rx_gain(self, rx_gain):
        self.rx_gain = rx_gain
        self.uhd_usrp_source_0.set_gain(self.rx_gain, 0)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.qtgui_freq_sink_x_0.set_frequency_range((self.freq-self.samp_rate/4), self.samp_rate)
        self.uhd_usrp_sink_0.set_center_freq(self.freq, 0)
        self.uhd_usrp_source_0.set_center_freq(self.freq-self.samp_rate/4, 0)

    def get_duc_actual_taps(self):
        return self.duc_actual_taps

    def set_duc_actual_taps(self, duc_actual_taps):
        self.duc_actual_taps = duc_actual_taps

    def get_ddc_actual_taps(self):
        return self.ddc_actual_taps

    def set_ddc_actual_taps(self, ddc_actual_taps):
        self.ddc_actual_taps = ddc_actual_taps




def main(top_block_cls=radio_ax100, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
