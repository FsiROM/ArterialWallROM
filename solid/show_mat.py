import vtk
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from solid_rom import solid_ROM
import random

fom_pressure = np.load("../nln_elastic_law_strstrain_nln_veloc/pressure.npy")
fom_section = np.load("../nln_elastic_law_strstrain_nln_veloc/diameter.npy")
fom_velocity = np.load("../nln_elastic_law_strstrain_nln_veloc/velocity.npy")
iters_fom = np.load("../nln_elastic_law_strstrain_nln_veloc/iters.npy")

idl = int(np.cumsum(iters_fom)[180])
prs_DATA = fom_pressure[:, :idl]
sec_DATA = fom_section[:, :idl]
prs_DATA = np.load("../nln_elastic_law_strstrain_nln_veloc/pres_TRAIN_DATA.npy")
sec_DATA = np.load("../nln_elastic_law_strstrain_nln_veloc/sec_TRAIN_DATA.npy")

ids_ = np.arange(0, prs_DATA.shape[1])
#TR_ids = np.array(random.sample(range(prs_DATA.shape[1]), int(.8*prs_DATA.shape[1])))
TR_ids = np.load("TR_ids.npy")
TST_ids = np.setdiff1d(ids_, TR_ids)

TR_load_data = np.delete(prs_DATA, TST_ids, 1)
TR_disp_data = np.delete(sec_DATA, TST_ids, 1)

TST_load_data = prs_DATA[:, TST_ids]
TST_disp_data = sec_DATA[:, TST_ids]
errors = np.empty((10, 10))
r_disps = np.arange(1, 11)
r_pres = np.arange(1, 11)
    
for j in range(len(r_disps)):
    for i in range(len(r_pres)):
        sol_rom = solid_ROM()
        sol_rom.train(TR_load_data, TR_disp_data, quad_ = False, ridge = False,
                                    kernel='thin_plate_spline', degree = 1, norm_regr = True, rank_disp=r_disps[j], rank_pres=r_pres[i],)

        pred_sec = sol_rom.pred(TST_load_data)

        errors[i, j] = (np.linalg.norm(pred_sec - TST_disp_data, axis = 0).mean())
        print("\n ", errors[i, j], "\n")

import matplotlib.colors as colors
import matplotlib.ticker as ticker

fig, ax  = plt.subplots(figsize=(7.5, 7.5))
a = ax.matshow(errors, cmap = 'PuBu', norm = colors.LogNorm(),)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

ax.set_xticklabels((r_disps-1).astype(str), fontsize = 18)
ax.set_yticklabels((r_pres-1).astype(str), fontsize = 18)

plt.colorbar(a, )
plt.xlabel("Number of section modes", fontsize = 18)
plt.ylabel("Number of pressure modes", fontsize = 18)
plt.tight_layout()
plt.savefig("mat_.pdf", bbox_inches = 'tight')

np.save("TR_ids.npy", TR_ids)


fig2, ax1 = plt.subplots()
ax1.semilogy(r_pres, errors[:, 8], 'x-', color = 'k');
plt.xlabel("Number of pressure modes");
plt.ylabel("Mean L2 error");
plt.savefig("fixed_disp.pdf", bbox_inches = 'tight');

