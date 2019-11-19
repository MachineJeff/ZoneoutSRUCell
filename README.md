# ZoneoutSRUCell
SRUCell with Zoneout, imitated from ZoneoutLSTMCell.

## Usage

you can use this cell just like the RNNCell

```python
cell = ZoneoutSRUCell(
    num_units = 1024, 
    is_training = is_training,
    zoneout_factor_cell = 0.2,
    name = 'ZoneoutSRU')
```

