#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: AX100 Radio Link
# Author: Shu Luo
# Copyright: CalgaryToSpace
# GNU Radio version: 3.10.6.0

from packaging.version import Version as StrictVersion
from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import audio
from gnuradio import blocks
from gnuradio import digital
from gnuradio import filter
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import gr, pdu
from gnuradio import network
from gnuradio.qtgui import Range, RangeWidget
from PyQt5 import QtCore
import math
import numpy as np
import satellites.components.deframers
import sip



class radio_ax100_icom(gr.top_block, Qt.QWidget):

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

        self.settings = Qt.QSettings("GNU Radio", "radio_ax100_icom")

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
        self.samp_rate = samp_rate = 48000
        self.baud_rate = baud_rate = 2400
        self.tx_again = tx_again = 0
        self.sps = sps = samp_rate/baud_rate
        self.rx_again = rx_again = 10

        ##################################################
        # Blocks
        ##################################################

        self._tx_again_range = Range(0, 40, 0.1, 0, 200)
        self._tx_again_win = RangeWidget(self._tx_again_range, self.set_tx_again, "Tx Audio Gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._tx_again_win)
        self._rx_again_range = Range(0, 40, 0.1, 10, 200)
        self._rx_again_win = RangeWidget(self._rx_again_range, self.set_rx_again, "Rx Audio Gain", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._rx_again_win)
        self.satellites_ax100_deframer_0 = satellites.components.deframers.ax100_deframer(mode = "ASM", scrambler = "CCSDS", syncword_threshold = 1, options="")
        self.qtgui_time_sink_x_0_0_1 = qtgui.time_sink_f(
            baud_rate, #size
            baud_rate, #samp_rate
            "", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0_0_1.set_update_time(0.10)
        self.qtgui_time_sink_x_0_0_1.set_y_axis(-1.2, 1.2)

        self.qtgui_time_sink_x_0_0_1.set_y_label('Rx Baseband', "")

        self.qtgui_time_sink_x_0_0_1.enable_tags(True)
        self.qtgui_time_sink_x_0_0_1.set_trigger_mode(qtgui.TRIG_MODE_AUTO, qtgui.TRIG_SLOPE_POS, 0.5, 0.01, 0, "")
        self.qtgui_time_sink_x_0_0_1.enable_autoscale(False)
        self.qtgui_time_sink_x_0_0_1.enable_grid(True)
        self.qtgui_time_sink_x_0_0_1.enable_axis_labels(True)
        self.qtgui_time_sink_x_0_0_1.enable_control_panel(False)
        self.qtgui_time_sink_x_0_0_1.enable_stem_plot(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [0, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [0, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_time_sink_x_0_0_1.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_sink_x_0_0_1.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0_0_1.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0_0_1.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0_0_1.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0_0_1.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0_0_1.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_0_1_win = sip.wrapinstance(self.qtgui_time_sink_x_0_0_1.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_0_0_1_win)
        self.qtgui_time_sink_x_0_0_0 = qtgui.time_sink_f(
            (samp_rate//10), #size
            samp_rate, #samp_rate
            "", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0_0_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0_0_0.set_y_axis(-1.2, 1.2)

        self.qtgui_time_sink_x_0_0_0.set_y_label('Tx Baseband', "")

        self.qtgui_time_sink_x_0_0_0.enable_tags(True)
        self.qtgui_time_sink_x_0_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_0_0_0.enable_autoscale(False)
        self.qtgui_time_sink_x_0_0_0.enable_grid(True)
        self.qtgui_time_sink_x_0_0_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0_0_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0_0_0.enable_stem_plot(False)


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
                self.qtgui_time_sink_x_0_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_sink_x_0_0_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0_0_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0_0_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0_0_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0_0_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_0_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0_0_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_0_0_0_win)
        self.qtgui_time_sink_x_0_0 = qtgui.time_sink_f(
            baud_rate, #size
            baud_rate, #samp_rate
            "", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0_0.set_y_axis(-1.2, 1.2)

        self.qtgui_time_sink_x_0_0.set_y_label('Rx Baseband', "")

        self.qtgui_time_sink_x_0_0.enable_tags(True)
        self.qtgui_time_sink_x_0_0.set_trigger_mode(qtgui.TRIG_MODE_AUTO, qtgui.TRIG_SLOPE_POS, 0.5, 0.01, 0, "")
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
        styles = [0, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [0, -1, -1, -1, -1,
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
        self.top_layout.addWidget(self._qtgui_time_sink_x_0_0_win)
        self.pdu_pdu_to_tagged_stream_0 = pdu.pdu_to_tagged_stream(gr.types.byte_t, 'packet_len')
        self.network_socket_pdu_0 = network.socket_pdu('TCP_SERVER', '', '52002', 10000, False)
        self.mmse_resampler_xx_0 = filter.mmse_resampler_ff(0, (samp_rate/baud_rate/4))
        self.digital_symbol_sync_xx_0 = digital.symbol_sync_ff(
            digital.TED_GARDNER,
            (samp_rate/baud_rate),
            0.1,
            1.0,
            1.0,
            1.5,
            1,
            digital.constellation_bpsk().base(),
            digital.IR_MMSE_8TAP,
            32,
            )
        self.digital_chunks_to_symbols_xx_0 = digital.chunks_to_symbols_bf([-1, 1], 1)
        self.blocks_wavfile_sink_0 = blocks.wavfile_sink(
            'C:\\Users\\ocat\\Desktop\\ax100_down_2400bps.wav',
            1,
            samp_rate,
            blocks.FORMAT_WAV,
            blocks.FORMAT_PCM_16,
            False
            )
        self.blocks_unpack_k_bits_bb_0 = blocks.unpack_k_bits_bb(8)
        self.blocks_repeat_0 = blocks.repeat(gr.sizeof_char*1, 4)
        self.blocks_multiply_const_xx_0_0 = blocks.multiply_const_ff(10**(rx_again/20), 1)
        self.blocks_multiply_const_xx_0 = blocks.multiply_const_ff(10**(tx_again/20), 1)
        self.blocks_message_debug_0 = blocks.message_debug(True)
        self.audio_source_0 = audio.source(samp_rate, '', True)
        self.audio_sink_0 = audio.sink(samp_rate, '', True)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.network_socket_pdu_0, 'pdus'), (self.pdu_pdu_to_tagged_stream_0, 'pdus'))
        self.msg_connect((self.satellites_ax100_deframer_0, 'out'), (self.blocks_message_debug_0, 'print'))
        self.msg_connect((self.satellites_ax100_deframer_0, 'out'), (self.network_socket_pdu_0, 'pdus'))
        self.connect((self.audio_source_0, 0), (self.blocks_multiply_const_xx_0_0, 0))
        self.connect((self.audio_source_0, 0), (self.blocks_wavfile_sink_0, 0))
        self.connect((self.blocks_multiply_const_xx_0, 0), (self.audio_sink_0, 0))
        self.connect((self.blocks_multiply_const_xx_0_0, 0), (self.digital_symbol_sync_xx_0, 0))
        self.connect((self.blocks_multiply_const_xx_0_0, 0), (self.qtgui_time_sink_x_0_0_1, 0))
        self.connect((self.blocks_repeat_0, 0), (self.digital_chunks_to_symbols_xx_0, 0))
        self.connect((self.blocks_unpack_k_bits_bb_0, 0), (self.blocks_repeat_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0, 0), (self.mmse_resampler_xx_0, 0))
        self.connect((self.digital_symbol_sync_xx_0, 0), (self.qtgui_time_sink_x_0_0, 0))
        self.connect((self.digital_symbol_sync_xx_0, 0), (self.satellites_ax100_deframer_0, 0))
        self.connect((self.mmse_resampler_xx_0, 0), (self.blocks_multiply_const_xx_0, 0))
        self.connect((self.mmse_resampler_xx_0, 0), (self.qtgui_time_sink_x_0_0_0, 0))
        self.connect((self.pdu_pdu_to_tagged_stream_0, 0), (self.blocks_unpack_k_bits_bb_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "radio_ax100_icom")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_sps(self.samp_rate/self.baud_rate)
        self.mmse_resampler_xx_0.set_resamp_ratio((self.samp_rate/self.baud_rate/4))
        self.qtgui_time_sink_x_0_0_0.set_samp_rate(self.samp_rate)

    def get_baud_rate(self):
        return self.baud_rate

    def set_baud_rate(self, baud_rate):
        self.baud_rate = baud_rate
        self.set_sps(self.samp_rate/self.baud_rate)
        self.mmse_resampler_xx_0.set_resamp_ratio((self.samp_rate/self.baud_rate/4))
        self.qtgui_time_sink_x_0_0.set_samp_rate(self.baud_rate)
        self.qtgui_time_sink_x_0_0_1.set_samp_rate(self.baud_rate)

    def get_tx_again(self):
        return self.tx_again

    def set_tx_again(self, tx_again):
        self.tx_again = tx_again
        self.blocks_multiply_const_xx_0.set_k(10**(self.tx_again/20))

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps

    def get_rx_again(self):
        return self.rx_again

    def set_rx_again(self, rx_again):
        self.rx_again = rx_again
        self.blocks_multiply_const_xx_0_0.set_k(10**(self.rx_again/20))




def main(top_block_cls=radio_ax100_icom, options=None):

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
