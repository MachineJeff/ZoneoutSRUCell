# -*- coding: utf-8 -*-
'''
@Author: yichao.li
@Date:   2019-11-19 19:16:40
@Last Modified time: 2019-11-19 19:17:20
@Description: SRUCell with Zoneout
'''
import tensorflow as tf

class ZoneoutSRUCell(tf.nn.rnn_cell.RNNCell):
    '''Wrapper for tf SRU to create Zoneout SRU Cell
    '''
    def __init__(self, num_units, is_training, zoneout_factor_cell=0., name=None):
        zm = zoneout_factor_cell
        if zm < 0. or zm > 1.:
            raise ValueError('provided Zoneout factors are not in [0, 1]')

        self._cell = tf.contrib.rnn.SRUCell(num_units, name=name)
        self._zoneout_cell = zoneout_factor_cell
        self.is_training = is_training

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        '''Runs SRU Cell and applies zoneout.
        '''
        output, new_state = self._cell(inputs, state, scope)

        prev_c = state
        new_c = new_state

        #Apply zoneout
        if self.is_training:
            #nn.dropout takes keep_prob (probability to keep activations) not drop_prob (probability to mask activations)!
            c = (1 - self._zoneout_cell) * tf.nn.dropout(new_c - prev_c, (1 - self._zoneout_cell)) + prev_c
        else:
            c = (1 - self._zoneout_cell) * new_c + self._zoneout_cell * prev_c

        new_state = c

        return output, new_state