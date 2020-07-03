from hoplite import Hoplite
import numpy as np

output = np.array(
    [
        [[9, 9, 9, 9], [0, 0, 0, 0], [9, 9, 9, 9]],
        [[9, 9, 9, 9], [0, 0, 0, 0], [9, 9, 9, 9]],
    ]
)

print("row: {}".format(Hoplite.consec_row(output)))
print("col: {}".format(Hoplite.consec_col(output)))
print("chan: {}".format(Hoplite.consec_chan(output)))

# row should be what chan is
# col should be row
# chan should be col
