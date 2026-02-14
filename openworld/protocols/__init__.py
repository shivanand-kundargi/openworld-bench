"""
Cross-setting evaluation protocols.

Protocols:
- da_on_cl: DA methods evaluated in CL setting
- da_on_dg: DA methods evaluated in DG setting
- dg_on_cl: DG methods evaluated in CL setting
- dg_on_da: DG methods evaluated in DA setting
- cl_on_da: CL methods evaluated in DA setting
- cl_on_dg: CL methods evaluated in DG setting
"""

__all__ = [
    'DAonCLProtocol',
    'DAonDGProtocol',
    'DGonCLProtocol',
    'DGonDAProtocol',
    'CLonDAProtocol',
    'CLonDGProtocol',
]
