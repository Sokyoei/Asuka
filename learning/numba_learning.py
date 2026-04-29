# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %%
from Ahri.Paladin.utils import timer
from numba import jit

# %% [markdown]
# [Numba](https://numba.pydata.org/)


# %%
@timer
def python_add(count: int = 10000000):
    a = 0
    for i in range(count):
        a += i
    return a


@timer
@jit(nopython=True)
def numba_add(count: int = 10000000):
    a = 0
    for i in range(count):
        a += i
    return a


print(numba_add())
print(numba_add())  # 第二次会编译
print(python_add())

# %%
