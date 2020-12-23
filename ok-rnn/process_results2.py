import my_txtutils
import numpy as np
import math
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# product = reduce((lambda x, y: x * y), [1, 2, 3, 4])

with open('vloss.json') as f:
    vloss = json.load(f)
vloss = json.loads(vloss.replace("'",'"'))

plt.figure()
plt.plot(vloss['vloss'])
plt.plot(vloss['tloss'])

with open('vloss_main.json') as f:
    main = json.load(f)
main = json.loads(main.replace("'",'"'))
plt.plot(main['acc'])
plt.figure()
plt.plot(main['loss'])
