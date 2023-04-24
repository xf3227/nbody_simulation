import ctypes
import numpy as np

__all__ = [
    'System', 'sys_rand', 'sys_waltz', 'sim_cpu_v00', 'sim_cpu_v01',
    'sim_cpu_v02', 'sim_cpu_v03'
]

_libc = ctypes.CDLL('./c/lib/nbody.so')

# define the System struct
class _System(ctypes.Structure):
    _fields_ = [
        ('N', ctypes.c_long),
        ('m', ctypes.POINTER(ctypes.c_double)),
        ('s', ctypes.POINTER(ctypes.c_double)),
        ('v', ctypes.POINTER(ctypes.c_double)),
        ('a', ctypes.POINTER(ctypes.c_double))
    ]

class System:
    def __init__(self, N, m, s, v, a):
        self.N = N
        self.m = m
        self.s = s
        self.v = v
        self.a = a
        self._sys = _System(
            N, 
            m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

    @property
    def _as_parameter_(self):
        return self._sys

# define the sys_rand function
_libc.sys_rand.argtypes = [
    ctypes.c_long,
    ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double
]
_libc.sys_rand.restype = _System

def sys_rand(N, m_min, m_max, s_min, s_max, v_min, v_max):
    sys = _libc.sys_rand(N, m_min, m_max, s_min, s_max, v_min, v_max)
    m = np.ctypeslib.as_array(sys.m, (sys.N,))
    s = np.ctypeslib.as_array(sys.s, (sys.N * 2,))
    v = np.ctypeslib.as_array(sys.v, (sys.N * 2,))
    a = np.ctypeslib.as_array(sys.a, (sys.N * 2,))
    return System(N, m, s, v, a)

# define the sys_waltz function
_libc.sys_waltz.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
_libc.sys_waltz.restype = _System

def sys_waltz(m0, m1, r):
    sys = _libc.sys_waltz(m0, m1, r)
    m = np.ctypeslib.as_array(sys.m, (sys.N,))
    s = np.ctypeslib.as_array(sys.s, (sys.N * 2,))
    v = np.ctypeslib.as_array(sys.v, (sys.N * 2,))
    a = np.ctypeslib.as_array(sys.a, (sys.N * 2,))
    return System(2, m, s, v, a)

# define the sim_cpu_v00 function
_libc.sim_cpu_v00.argtypes = [
    _System, _System, ctypes.c_double,
]

def sim_cpu_v00(
    sys: System, 
    buf: System,
    delta: float
):
    _libc.sim_cpu_v00(sys._as_parameter_, buf._as_parameter_, delta)

# define the sim_cpu_v01 function
_libc.sim_cpu_v01.argtypes = [
    _System, _System, ctypes.c_double,
]

def sim_cpu_v01(
    sys: System, 
    buf: System,
    delta: float
):
    _libc.sim_cpu_v01(sys._as_parameter_, buf._as_parameter_, delta)

# define the sim_cpu_v02 function
_libc.sim_cpu_v02.argtypes = [
    _System, _System, ctypes.c_double,
]

def sim_cpu_v02(
    sys: System, 
    buf: System,
    delta: float,
):
    _libc.sim_cpu_v02(sys._as_parameter_, buf._as_parameter_, delta)

# define the sim_cpu_v03 function
_libc.sim_cpu_v03.argtypes = [
    _System, _System, ctypes.c_double, ctypes.c_long,
]

def sim_cpu_v03(
    sys: System, 
    buf: System,
    delta: float,
    step: int,
):
    _libc.sim_cpu_v03(sys._as_parameter_, buf._as_parameter_, delta, step)

