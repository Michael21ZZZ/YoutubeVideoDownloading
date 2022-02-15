#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install --upgrade google-cloud-videointelligence')


# In[ ]:


import os, io
import pandas as pd
from google.cloud import videointelligence
from collections import namedtuple
import plotly.graph_objects as go
import plotly.express as px
import requests
import string
import proto
import json

## download your API Key from your Google Console 
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/content/drive/MyDrive/Colab Notebooks/youtube-video-analysis-333605-d50218259c5d.json'
video_client = videointelligence.VideoIntelligenceServiceClient()


# In[ ]:


import difflib

lines1 = '''
-0x8gTye-AI
-0x8gTye-AI
-6p4H3jmF68
-71ld0iqAq8
0-6lxT2HaAY
0-JON5_KYYI
03zVNKUwM98
0JA8yHj76_M
0Kuvp_AleAI
0VENoaBKKIE
0VerMp1iFhk
0Zom-VVNqmI
0bRS-6GDLy4
0cyZNGBpUHA
0d4H7xd_F1I
0gW5qlc0Rtg
0iS_3uQSXNM
0kZC6233ryM
0mbODbpXpDA
0p0df--IQUg
0pbyQhHWh28
0rdUeoKfcv8
0sz3eC8jW0A
0uQMLQUC4iM
0vyLi8BVi8I
0wUhspJgBcg
0y93_JOsjBM
0zoAAA7-Lng
1ABNwZRlLpU
1C2WpZA_L4E
1CsiD-ppAaA
1DQEbZ7g7u0
1FvyCw4wQC8
1JS5Y9BGWT8
1NIhv6fCqAU
1TTjOMYZEHQ
1WrdzNJ4A44
1bflUzy_2Y4
1dH5NbVgJHk
1hHk52k-bRY
1kfwzCmYbbk
1lwYseaL4m8
1mtGB6HR7S8
1nl2QdZY2X0
1osIAtrto2k
1qcMSTMZ6ok
1qj5tOq6EgY
1s9NcgkuhXE
1teEi1piQrM
1zxUExdIN_4
21l4imBvLyM
22Ty9QW1x5A
2A4DCpO10-A
2ASbKuqCcXY
2DgCpZOdskI
2DqwNT5qe60
2F6EkgAyuRA
2H4TxQIqYM0
2N64oGaFQCA
2PqJyqMZ0Eg
2PzOkiRDXPE
2QpIUlv0e28
2TYyRyqesyg
2TdclW7RzSY
2UCvs-Lw4rA
2UnLgXCpCEg
2WMITcgSHvM
2a6Ksm1OTEU
2f2j_OwVMKg
2jK3W_RSEwE
2kZFHoQyg-E
2oyZ7QyJTB8
2sWo_eab3Vk
2wUawmvw_0M
31jNN10-Cmk
32F2hQTncQ0
32fJ7wnRIRA
32xm_1KSWVA
36nUFQFm7HA
37ytFdg556M
3Lo6MKc0dIM
3VTNWvtkob8
3XtPggjBmA0
3Xz6w2w2vfE
3ZX33fUL6t8
3c8E-iS990Q
3cet4bO40V4
3cihIb_sBeg
3gnkVs6dFKg
3ir7xA1JlKM
3jPGl_nADAA
3ldWeYEGYdM
3nFcU-EODu4
3oJV4kGgy4Y
3sUQ_jBDJXI
3tQ0eB1Ay4g
3zbH-FB0cUI
42PTZv4WRi0
4F2dGseG6NI
4HrGpWlT9GA
4MvY8VJfH_8
4NgJbp7K6Hk
4Vc3yKgcUMQ
4X-XwLNtLgE
4Zy40_BAGdY
4cQpHG-a4SI
4hWWFqVDAIE
4rS3CWZOeE4
4s5KO1n97GU
4uQrCTkPrFU
4udV2Ai-Kgk
4yjxdO6ObKM
4zk-UQSOrWQ
50yRwPa537A
52kEHcp5PKA
52sEiZzYP0I
57ySNIcvCTg
596LMJTJFsk
5BZKrlxBREM
5HNFsjUwolI
5I-Wr_s00aU
5KWAgKR9JBE
5PZTqQrvSck
5Q-XeVxthOk
5RXRr8PKunk
5UkXnEjv7x4
5XO6-HrTajM
5ZNpKymvyo8
5hXI7YJt8o8
5kEJl7hdIaQ
5lLQtbBp6yk
5n0QmwX62QQ
5nrjL5FCg1Y
5pV9DjRBIoI
5q57HFAkC4g
5sQn3f7w6A4
5to6u28pmcM
5uke089o-mQ
5wS3qtQyue4
60kSP5JgzwM
60mQQ2eKvaE
62OTciMk_wY
66j1tTlUgLU
67QojWkQLsA
6Ac3Vujv3WQ
6BG1_vfLdug
6C8bZgOnAw8
6Exj0n0VfN0
6I5J4Mf4gMk
6M4yHhuawII
6NFT31grPf0
6PxtWgIbKi8
6R1V-Mlr-ZY
6RYLs4H1ciE
6UBz3lm6ldg
6Wj1xH8AM7Y
6YHKIaxkW_g
6cMnR4PfJfQ
6g4CSh04KQg
6nGTOTGuLOg
6nSVHzkus7U
6qLzjy7Hr9g
6t4tBmbPko8
6wFcp6cFsCg
7-cGz2eLyb4
72sL_WEQedU
74s6eiBBBFU
753t24hUTrg
79CmTSZ_g5U
7Au-MRjoHYQ
7JTCe5n2zSg
7MEdFIXTHP4
7MvpkJaCwRA
7NCUAyTjUnU
7OXb_xekQY8
7TQ36plrNjU
7XWPdWhG_jY
7Xjxdh_6adQ
7ZUp3ddSYoQ
7c7_k56DiVY
7cKGVYT3W8g
7d6Sn0EOilI
7iIyA9QWopA
7lNlME-LDZY
7rZNgb_ncNE
7u8Od1nDs48
7zB5aLrLyAE
8-Qtd6RhfVA
80TrJa4ZlrQ
81SoA_cBKMM
829npgW0ceY
85_TFb48iw4
87TCC0iWB3Q
88N5kx7WjZY
8Fw70WGcK20
8JgZruMJ0fU
8L9w5vTlRRU
8Lxb917AXF4
8Rwnraf0sq8
8SBIXJSVdTQ
8VH9UINEI24
8YGfjhjcKY4
8d9jn7RNkcU
8ezyDlUF42A
8ftE8A4S6FE
8gH3F_A-974
8hTryXqp6g8
8hnvY6_Jpvg
8l4CKLqElOY
8mOjYFwmgxk
8sI5XaKs5bk
910K7B_1b6I
91ZqkjbtHFU
92gq7UYoau0
976SFoINOU0
99CLGZNuFqE
9Ax06n7V9VE
9IXcZ1G3d8I
9R5PY3IDaTU
9TdjoZKXKT4
9TmCyAjaxs8
9UVl4HQ2V1Q
9Z50YyMFCaI
9Z9eqr8sNIM
9Zr4wG3ToHY
9_8t_HSG-uE
9abusStvqCA
9acmEoE7dOk
9b1eBqIoR7I
9ccIlIId_R0
9fBLYpYwyek
9ghdculyWY0
9hIxfE8gDYk
9jkDqITkXIY
9mWX6bkgY00
9o1CIJJFV6M
9rQhskjdLXM
9uSSBu-5eSY
9zb0Oo0ryEI
ABeOnnq6k8M
AF-Qlsd2c8Y
AFxN0BNd-D8
AJ9QNdP9fHk
AMHe5wrNguU
AMYHNvBAQxI
AR_rgqo1xYg
ASQo9bIPch4
ASysfc-IYw8
AWuz9V0SH-M
A_LjUmoT8QI
A_vg1AT1A-Y
Aa3ycuHbykQ
AjDp9MClUY4
AoHvkrcr6EM
AqS3Pms8M-U
AqV5bUpbCpY
AumWr7lU0mM
AyjYSkr6sRk
Az9lWdqebaU
B1m-GQotG-g
B2-kRFave8Y
B4FcRYPlDOQ
B5h8o1HZWmM
B6GQfJmAf0A
BCSdVZtWOaM
BFUis11qaQU
BGwKsNL22NA
BI8WvLUg8K8
BIOk_BfXUzk
BOSkQ5MnjT0
BVUSo_hs1Ag
BZOiyQkMOS8
BZS0amkjOhI
BgNpATWfR5Y
BnaCRU3vjTU
BodnYcHGtiA
Br2Iqs6vWsQ
BsMVEFmwFwA
BwPhcLw5B0M
BwQsh1TDSMM
Bxo9YgIpr90
C4JRCFLKKHQ
C6w4s5gwaLI
C92dVS8NIw4
CG9y2uXXFcU
CJcfcj4C5-U
CJeKpR0oui0
CKlSFlnmDcY
CLg-WWagkKk
CM1u29BeqC0
CQVg8Tv5yQ0
CRWhQuIxspA
CWyrp3hu4KE
CXRtk0E7aN4
CYGLevDEj08
CZMj9iRTAI0
Ca9mh353EDY
CbixFksNU48
CdXWpvps_pg
CfcQdHuVZuQ
Cg2ZSaVOHsg
CpQIcK-zetg
CrjjN2sXB_g
CtTkHszSo1E
Cvoonq7Resk
CzVscrIPr2Q
D0Ob4g8fgf8
D1CUNDwCQdY
D5OPXddiYcA
D7lenSSqbP4
DA_G8iK0bAI
DCNBymqdhEY
DIIjaKwDsf4
DLux6jxNrcY
DPxumNR7UtM
DQYBVdPPDsQ
DWA_cVm7mEM
D_6XgzowY5Y
D_vcr6dHV78
Dd7vMuw3ExU
Dendav4u2yk
DgrXeBJ66_g
Dgy05UCXLQ0
Dj1b8jUzmZk
Dnu8fmx4CI8
Dt_KmEfx5D4
DtnKKb_wtxI
DyGIvs9zrVA
Dykye9vhypU
E10WcB1qFVY
E2MC3Y7hR-Q
E2SkzWC4siU
E3nTHnkZybk
E5zwwtfNomY
E9MEZXpjrYg
EK7Fq2gkFrQ
EMsmL9NfkuI
ERbI9G2jBbk
EWqs0qdXeZ4
Ec6ICpkHAR8
Ej2_mj3Fyu8
EjFrczNSRE8
Ekqq6DE8vGE
EleUObRXQ0E
EltIpB4EtYU
EpEWPFtZjUI
EtC3WoFLxhE
EtH-OOIcykk
EwQ1tPF8MgQ
F2qgSTB8Dyk
F4Lt0AjMxEY
F5zmokcqM9g
F6lzCUQd-v0
F8UGIKR3okA
FA8iUBEDIS0
FArSFWosHq8
FMG5_iBd0Sg
FQgc6e_vMVM
F_AbJs9-EU4
FePjKBoFICA
FlVnh6hO7l4
FldsntzddF8
FwpVuQi9UQQ
FylbqN6XYBA
FyzzDtzfC0k
G11lr2N7gjQ
G7SOs1s-i54
GEHrIGP9M4I
GF-YNEGTaCA
GGgGjtM5ipg
GHVtiJ9xd6A
GKSp2Ogv564
GLd_x_WFaqk
GO40_Zi9WAQ
GYpU99BTbzg
Gazuoyf-lm4
GcztVMg3Z-k
GfL3GCFU-b0
Gfr364SyQcg
GgkBFkVas00
Gl9EsvAXgDY
GqoUzm8ej5k
GqyS_L71nEY
GvTta5ib0MY
GvbSiwZYh6M
GwEucKnTerM
GwQaekb5onQ
Gx6t_x0hF7U
H2IqRgAHpFk
H5LuUPA8Hvk
HB3m-Ni8jSM
HB5O-aqR3G8
HCeaUF9sCEE
HFT_bGnNLK4
HIz9E3qAfiU
HKP1VjQg6Jg
HL6w_Xw5C_g
HMLCN3DI7d8
HOwqo56BKQU
HQ0mIz0I14E
HUd9ucjaCLI
HZLtwCKwYYg
He5HE4y4vmE
Hrdu4Dh6DoU
HvcQdJcN6mk
HwFvwhwuJ1E
HyiSxfdytoA
I13xK_g27rk
I1WOEwRVjdg
I1hZSLXAVn0
I6G6NPWe2mU
IKCDefMkH58
IMK_XBI8Yx4
IMLVbEzM0DI
IMPzAdVOhKY
INC6bICXmKI
INpDGmujLSQ
IO4Gyto7KrM
IO6XF9RPjC4
IRV2wWm6BQk
IVH_6ciQfZk
IWPE3Cdnc34
IY699mzi7K4
IZZaeo4jBBM
Ibr429pU7og
Ig8JNw-Znt0
IoBDTd9vHJE
IpPHCLrEmSI
ItjPn8WxVSA
Iu9GPd4A-nk
J0Fd5vJGAWU
J0R1p5suhtQ
J89Qw3MOIo8
J97G6BeYW0I
JASPk6dLvKY
JAjZv41iUJU
JAs2K6mQSmM
JBjEKPOKlVE
JC7AiQmAynE
JCQk7kVcvUg
JDSrcVa4aS0
JEZvmDtj4UI
JJoyWK58CvY
JKMuJy12K-U
JMknA_CCnW4
JNKaTZIK3NM
JPJ4l7QZ9eM
JSFiOF7xGfE
JTtSyFi2zB4
JVUUCE7aluM
JW3JyccEXRI
JWgNUPI-dMI
JXygbkEjNIE
JbDvSPJofeg
JevaHIjSK_g
Jl77C9JQa1I
JloyNp6P7oQ
JmHkHHkiGwk
Jn0umG_ZuII
Jn8Yuo20G78
JoJ13RWsLV8
JqJaVvLO748
JrhO9vyrztQ
K1Zg04lJXgM
K2ICK70lEr8
K2QqsZAoQwA
K4SOS9okas4
K4UhxeuSKw4
KDP239KYaPk
KFMs2HpFoWE
KFgr-elM9Hk
KJqRSDyfFvU
KLMnpgVBpJg
KM82rY3WEKM
KP6Zm9vl3FM
KPjvj0P-vAQ
KQVHtp609p8
KSMA1gYL4ZY
KW8HnpSp6wk
KYyMI0gyLYk
KfYBh97HU4M
KgL1jozNYGU
Kih-xzajeGI
KiouveG278Y
Kn3lgskKA08
KnArDKQqmj0
KtT7zvYJekA
KxQTLP2KeaE
KyPbLw3t_RM
L-a-dKhuYj8
L2ZNvl1saLY
L3r_pfmelVQ
L77b57erQ4M
L8lBrTXUVSg
LBpwiaWZJLE
LCdTfQXDK34
LHCJ-Xq-Gak
LIuep2DazL4
LJb_bqfzuxE
LJvQxGd6uHM
LLU3zNPpvEg
LSUUsAcUSfE
LSvWz4Iul2Y
LX9iblqBfXA
L_gPzOxZ3TE
LadqiXXBnl0
Lfahe1cmO6Y
LniJbcPpxLM
LqSBrVbVBVQ
LrtiKvSd2r8
LruiDyDmzP0
LvaVsnHbIbI
LwLWo6kUst4
LxpbOl5TipY
LxsUZUI0MkI
M1bQ-gKzd1Q
M1dA33-Jvdk
M4BOOuJ2oiI
M8BWBrhaeKI
MG-YQAWJeFk
MJMG4-VH_rI
MQ1dhCmTpVg
MRJlb24X7X4
MUw11H-aCMY
MYQzJKeYP_M
Md52nIb2eb4
Md7PssTAvz8
MdM3I-iqyq8
Mdo5wdQNa_g
MmBmhxyb0t8
MmQzk3f1fKQ
Mp1ADUs1vNM
Mrzn1MP2Mb8
MsGmoVknaZQ
MwLl9q9VEu8
N-KZR7rxSq8
N0fiGtDcKPU
N2CeF78AGzU
N3kL6SKSoYY
N6oO9n6t8s8
N97-kDLucyU
NBnnwiyuI4I
NI79s7lcW-4
NLCaT3c7J3I
NMvTYhPnL8A
NQXGXfpV9d0
NRH-PJoaP58
NYm5sB6rT_E
N__Fekod1fY
Na-HWnYIy4M
NgX3JS-HnB4
Nnb4jrQbHyg
NqVXPZs66k8
Nr_arsIE7vA
NuzV3h2MyN4
Nvd5uk5z24A
NvghNK7TRjY
NwBzGfKY_gE
O2nAICrb7Ck
O5hUR5wPFHA
O5o5EC7fOaA
ODM4I95VxME
OHn2YUQJ98Q
OIM_5PBEU-E
OLVAUYDNbpk
ONZUfwAdHKE
OOBBD_5_KaY
OOI-raXimT4
OZ27w3gWDN0
O_kXOnrYYRA
OdHj5qimfl0
OgpVYTnsmuk
Oh_mPlJybbk
OlIVt-Yv6I4
Ov65fQWy5PQ
OvyUoZ9K8R8
OzK6zn-falA
P1DJRdHUDw0
PAab0M_6fuQ
PBB4SHQHTbY
PBlRQTRAc-Q
PEF5rIneraQ
PFtxU_0TNpQ
PGWDP-846Rg
PHGmEK7_49M
PLoNvGq1by4
PMEZ7bRnHwg
POyEK_C4XP8
PPpLxfSvT9Q
PTEfGjlVp2E
PbpuctabwvM
Pdec8clsV7Q
PfX-aCuxfBk
Pl28IILPDTU
PpC1cE7ZeEQ
PqgKFsK7f-Q
Pr_-cUA_PY8
Py9b7iSl5w4
Q-n-BRvSaK0
Q1aHsYftvLI
Q2AleRgc_rQ
Q4p6Suko25A
Q86xxuh2V2g
Q95gTsdk294
QAT_a8Mhyu0
QE0sj2hwpYg
QHIxH3wbc84
QHe9ReWwkhk
QJmBbhjFP6Y
QKdOYRneU44
QMH4dkyDL90
QPj6e1aqiB0
QQy4vasH4B4
QStFQDB4Dt4
QZu4UYrqoUE
QfrwYsrqo8I
QjMIA-vaLaQ
QmH-cLBdsRE
QpKBRC_RJFM
QqQ4KZ3oT74
QuSQg5ZVbp4
Qx4G7pcx0yk
QzhdLt9e2Ck
R0lq_XM4NHk
R3KuHxrtrg0
R5k57QaVB-0
R8iGRccartA
RAljwauggrE
RHZoQcQozYM
RIOPPpmIEo0
RMs7NvrQx8Y
ROgY6F0oRq0
RQJerxixvEk
R_gp8I-ZyZQ
RaNNNrmcJgY
Rd8ohUO2-4M
Rem9ZfiJ8wU
RsytUXscl4A
RtwI2JocwO8
RycF0ub2Al0
S-3nfpc_GiQ
S09YWIykt2Y
S1pznvqNV1U
S5ybbawRqbg
S8__zp7PdZM
S8ncSP8ozvg
S9Yw8WoTT-U
SA7yCKxzzr0
SG7NjIq6TV4
SHmjM_EHrzM
S_nJNks8VTI
SbwAB3vscnA
ScjyYj7UdsM
Se5zpcXtefY
Sev0xmmU6LI
St9r4pc2QmE
SxLTYDKWSrg
T207B-r6vYM
T2wKkoSDqWE
T5sqFzCvswg
T6lKxkzSnSY
T7J-yADVRY0
T8rCQh9t17k
TAwycgsaMlQ
TKnlUbM37yg
TQhtAjJikUY
TRUhLSURFO0
TSf3bS6ry-8
Ta2JyiazUtg
TiwW0AMIRiM
TjxVyLTu5vI
Tljta64txY4
Ts2qo44aNi4
Tt3J0flYZt8
Tuybdi-iVF8
U39pKokdcsE
U66C5FlolYU
UD70qaaI4E8
UEaUs3YXlt4
UGvSXyNJAAE
UHt5LXLncvE
UOvWr-vMO94
UPhGGR_Xyo4
UZ9JbmQiqYM
Ub3iwsiI7GM
Uk6U9ROn_MU
UphNkxbNC-U
Utskmy7lEu8
UxugWPhL_mU
Uxw75plvyx0
Uzj1vex_H24
V3D4Uz_rH5U
V6jFEq1T-wQ
V9FDxlZ_sik
VJgnv314808
VJkYLU2seF0
VNqmDmYMilU
VStR7iOeR0E
VTvcSqAVX5Y
VUqG6Z1wx9U
VXAFdiRubCI
VYLpIuUV-LQ
VZBGZ7YT-vY
V_DbpHyyzng
VbwRYFMPZS4
Vc_xioDS1DQ
VcltNvduSX0
VguGd-Tav3U
VhvGCUbYLrE
ViMyWYbZmYU
Viep2gZx4qg
Vlg7J0Y1RfU
VpobvFPR6hQ
VsbCOtZYGzY
VslZoZD9njE
Vtb4xCFygiU
VwWGCukz7O4
VwbCKToosT0
Vzvf7SiOk78
W0KPwTy0W9k
W1_pSpqCS58
W2Hz-M917-4
W3xE6BRra2g
W7Fk8O88cuM
WBjchfwi1jA
WEM-Nc_dn-Q
WEmzGVoYHO8
WF-I9BA5F00
WIpxrtf7QbY
WJbZg1PgzO4
WM3v85H1jX0
WOPcKqhV-o4
WPgsgZ58jW4
WQtwfwmYpsY
W_eG_bDhD8g
WcfONpA14iU
WjyQTmJBXFg
Wmtf2SPr3-g
Wqwlx0LrsVA
WsrS9XszuN0
Wtof98E8cC4
X0LUf1PpuPg
X0s_uUogolQ
X2rOWkOcq9I
X87NbBTXixE
X8G1YkurALI
XFBYZK_JCiM
XN3grZJUjUY
XOHO-j4ay6g
XR7Cy9hoBPA
XTvwESqfIAE
XUIktTOH2BY
XVvHbi0bZPM
XWhQlwaKY7Q
X_aNjxWOocA
X_yF9QFErpA
Xe_LsrAKP6U
XfSWO8a1PlQ
XfyGv-xwjlI
Xh6Vs-NiVTI
Xhd368LZRKA
XiQdXbkGwkI
Xz1LmdofZoQ
Y3iFuNgJbcg
Y5AL8FruOJE
Y98ry3671Rg
Y9tD5qMeDnI
YA91m_--WAg
YCsyTd29i9Y
YEbm6yp6Wf0
YEvH2mPyxWI
YICvHsKmjzY
YR3rrZV8HVQ
YRmJqwW36nA
YXhzyf02peg
YYAr1h1Rxq4
YdkyfGBc9YI
Ydzo0ypudQM
Ye9SRh59pjg
YeDDPumPYNk
YeJI1jeg1n8
Yhn2qguQfUM
YiIPbFuuboY
YiQfe-oz2C4
Ymtiig6l3zA
YqYQgtpKXrM
YxTpeB3vb9g
Yz19HLEEXqg
Z-cxMdEvsZM
ZA8GzhFh_CQ
ZCpSiSS0Yv4
ZFlgzUwX2UY
ZFwcGVGNOzY
ZHVkA15HmbM
ZMAhHaLg0lQ
ZMWGfX2jBDA
ZTReTNm8abU
ZTqXRVGb0dk
ZVxbRnbFRB4
ZWrybTGDFHY
ZfSlQJk59Ng
ZiITCnnFbwQ
Zo9r9FRNILI
ZqVpqelyLFE
ZrkB236xr94
ZtVoHOayVmg
ZwQN4F8bLyk
ZwZigB_LmTg
_2m2OJgYxgI
_8VbgtVMf8Y
_9hPFlSnnNk
_A2Vo9kWdJI
_Byv0Zx9fHU
_C0i8IuHXEI
_CKuBMAlO2Y
_DGyROs4QD0
_GwKaHwd7CU
_JjAi11j_a4
_KatXIHOd4Q
_KndwmMEwXA
_N6AAre8Zag
_PQPw4bLrlE
_QPz0ui-7uM
_RALKmqnsYA
_ZUj2aYMDUA
_aLLrQVX3L8
_awmMMKmoS4
_bQi08rSz1g
_ex2PNXdhgM
_f-KFSSzzlk
_f9oZnhgp3U
_fCqEIn7ccA
_t7J9U_NFvE
_uXvihpw_KM
_utjrrpM4lk
_ygn-XrT150
_z3a81ge5AU
d5kVw6fx-n0
d6H_jPqjLKk
d8PmuZLlBqY
dC7y9qB3tcA
dCZ6d1fElpU
dEJ_r_03A68
dFtBokeHkWA
dH9Y_rby-jQ
dIqGm-Ib1oQ
dJ2F0RhvIhI
dJv7fFtPKwg
dK3Iyl5_n-Q
dKYxRDO2G_0
dL892eus81M
dNtwaasl2y0
dPKvHrD1eS4
dPyDyldwUlE
dQXKNXz_ZCg
dQr9asiswzA
dVxkV5KhTV4
dW-iPDFJtBU
daUGeFjO8NM
daVDStDLkDI
dafRjRtvfck
db2XgUgN94I
dhGvYjHzcIU
dhPZNnhZ5vU
dhrERViJSm8
djhv9FDyyIQ
dn0W0KJ4g48
dnRcZ5RMcbU
drAenNuBpPk
drXGjh3jRH0
du9qrh2DMck
duDHe8TGplk
dz6SNdKBnHI
f2HnkDGuBqk
f6txLUgJD54
fDjWczxy374
fF7dCRNQiek
fFRuvcYyHo8
fHEVfBzEANI
fHui_Q6ld10
fSYFsrurYKI
fTA5HOa87pM
f_O46jTHCtQ
f_bFPNbfNhQ
feNKdaMb1YU
fihS3YUlolY
fn2i9Z9TH7U
fojJ_H7hxkY
fsGqVYlUtbY
fw2vF05J4Z8
fxLagrFbPD0
fyOFbCmTzaQ
g-9YuxM5aNQ
g55rOPM-lWI
g56j1HhU0RI
g6J05QHC26Y
g7JLG36ZO-U
gANsmfNjEnA
gAuNCvWbRpQ
gC2cMr2fLgU
gFQF8i1Wmf4
gIZT0Ew0ugU
gJODa0YqzKg
gJfrTzkB4KI
gKd6_EqFI20
gMHbBr-dHJA
gMHiLBFNcNQ
gV326cyB0lg
gVro-wGpMpY
gciQEPUSnvI
geO2L3fzF5g
gfoDXFrih3A
gzwtHkUaOfk
i44Av6afijM
i5VhqPTHmv4
i8LUX2JEzyE
iAGkZmwB9XM
iIYqxBUM5dg
iJjhc5Lf744
iKKF5yuxvg8
iLllLj9n_Ng
iMzqR-XzueA
iN6u1fuJor0
iQEpDszK0Ck
iRgKZ-kHgOM
iSVEAqBVCwA
iS_S6ByJYLI
iUofnjY-HXI
iaqGLfsucJ8
icF2xJsSyX4
ie-oS8vQaXU
ifMbcPmdH6U
ik--FDrR-jM
ikGl7DPXUK0
imQ5qMqdoJM
is9Q5ZmuhqY
iyxhUqxusuY
j0QmUup77i0
j9oeKa-Brv4
jGpH3AD_Mj8
jHZt0naRnbM
jIZjtqUmSEM
jK-63yMLNk4
jOICrFbQGI8
jTPypnbFNFg
jUfGVwaB5UM
jWxMP8eK_a0
jYLh7_X2Cik
ja8T_rzO7F4
jiLKEjSi8cQ
jwD7bJrYH6U
jy0oXpqobwI
k1XG91UBQa8
k20meh6MRKU
k22zlWRzbk0
k42Ne6jiEIQ
kI6FITZycek
kIleZ3gCF3I
kK6mIPzBWWM
kLHRQJHFAqI
kMOQaxf9pag
kMaSLzivnWU
kMyx68jVRB0
kO2rU72dh1U
kOW2UfaVNzI
kOnN-Zh6HHE
kT8lFdOxVeE
kUaJ3OScyGY
kV4Y5itBXJI
kXP0THL-oO8
kYph5ulP2qA
kdbcA0K--d8
kefdajmcDzM
kfLEIN0X1Y8
khvpSnWYFJI
kiPd0ezjme4
kjMUz28ZItM
knXSucCKnmM
kp1hPho9RKA
krEeX5G9AYM
kzTG83mLRKU
lEutFrar1dI
lNAAzs4C-lQ
lOeQLJCVOdM
lPTlx2dcoNU
lP_kU6akp_A
lPvB3FqqigA
lVuuisHvptQ
lWQeanGvLUE
lZ2V_VfswRE
lZ_FN1Rk7zI
lchvWAnJSwY
lfdc3kG8354
lmq7wycWwu8
loUuq7rPTH0
lpHJP-oaLvI
lq4S9L1qjTg
lqLnrqV6I5I
lqeGgoPttAg
lrrEhc3SRJ0
lshzaYhtOyA
lu7-Pp36zWk
lxPU9pAClAA
n4h0_9u0EOM
n7H3KfMNpX4
n8KdfomQHkQ
nAFELKYkiuU
nClrDGP65A8
nHuO80WbrDI
nIQl9KaLfSQ
nJ0YMjp-c94
nLflozGJPAU
nSGIFs-s_UU
nSuBzZR6llY
nWUMAYb31rg
nWgf-odA50Q
nX-Ikn3hK6Q
nXCv9A2kJ64
nXyIaDgTBAk
n_yAt90fzFg
nazSmyMgYnY
nbV1seoE0KM
ncpDzJH5EoM
ngH9-eyRrhY
nlvQYF62j6A
noICBDh919I
nq20aTEWppk
nqWV5s7P8QY
numUKaH34rQ
nv0PfJMa95M
nveWpFwDTnk
nwSquF9eHck
nyRrm-cCj9g
q6D0bJXBS8Q
q7KObLnrIyM
q8BOimu8oNM
qAwdZBanT8g
qAwrjX0928Y
qG3OyONVbEQ
qHTURjMqh-E
qLGqcIFu9DA
qNdoOd11Vi8
qQarsq-1ykE
qY8694ZyjbE
q_t-KXiV0PI
qcmedjz_FpI
qcw0IuDr-28
qii0YXVkFjQ
ql0lY4qKBTU
qoy7XPz9XXE
qpfJTM8JvZE
qpzS1JD1Zkg
qusGm94uFJE
qvPEkkiAoY8
qvaztKvcasU
qx31aecUA5Y
qzhAGXJFFBQ
r1gt7XUUCp0
r2IYMx8lmp8
r2tXTjb7EqU
r59S0s0S7AE
rCErhY4VKCY
rEPw0DXpY2Y
rGlwJ0B5aOk
rH_bZsO24tk
rLjIJlz6ZAM
rMMpeLLgdgY
rPmDXv-fBRQ
rQXl6XT4BCE
rR4iKEIU0vE
rUsH1dHWUI4
rW0nHlwPWu0
rXE2CiCntGE
rXqGbNinhaA
rY_siujVJEg
rZr5KTurMXk
rasZGZpQsy0
rcOTAlcM89Y
rd6T_aTbppE
re6muxqRs7Y
rf1sAHA72uY
rgQ1ffvr5W0
rh1ZZXKW5UY
rhmAyFY6dhU
riDnpJihZ7k
rkfeOy88-_0
rlveYN26dLI
rnPIDHlxiH8
rpPeJFFzJ5g
rrAEg5wSa30
ruz7xePInwU
x2wXOLHojrU
x6eE_kkxXT8
x6jQw55IeWQ
x8zGj1UM2tQ
xCW_EFv6wC4
xGnAi4F0CG8
xJBqbv8GRZ4
xJVXL5oCZCQ
xSGpX3isqUE
xchGJ4UxCPE
xeRS9N1aV84
xfSL-OdZYbk
xgy5LJAGFbg
xi9wttwNN8k
xiZ9XlqhrBI
xm2T2LXZLtU
xyQVFAmrsh0
z-iZjn2VkBU
z0Pt-x8eKkM
z3TEPkzF32M
z3zkU8zf1b0
z4GO65w1t4A
z7WZ27g_r8E
z9aeU2Qw1S4
zEO-9wibPJg
zEyifMatos8
zFuoStIj_Y4
zKjwEImcdgA
zMwhbVhbXWA
zNLpPlqyS48
zOf7NCMa4ZA
zTUZ4-lQ394
zUrnx_LQiOg
zVtMGUzr4fA
z_mb9_FHdgI
zaDMhrzC8eQ
zbf0F1X0Z24
zjygvoYw4jg
zq_v8lS8uJk
zvmOwg9fchY
'''.strip().splitlines()

lines2 = '''
-0x8gTye-AI
-6p4H3jmF68
-71ld0iqAq8
0-6lxT2HaAY
0-JON5_KYYI
03zVNKUwM98
0JA8yHj76_M
0Kuvp_AleAI
0VENoaBKKIE
0VerMp1iFhk
0Zom-VVNqmI
0bRS-6GDLy4
0cyZNGBpUHA
0d4H7xd_F1I
0gW5qlc0Rtg
0iS_3uQSXNM
0kZC6233ryM
0mbODbpXpDA
0p0df--IQUg
0pbyQhHWh28
0rdUeoKfcv8
0sz3eC8jW0A
0uQMLQUC4iM
0vyLi8BVi8I
0wUhspJgBcg
0y93_JOsjBM
0zoAAA7-Lng
1ABNwZRlLpU
1C2WpZA_L4E
1CsiD-ppAaA
1DQEbZ7g7u0
1FvyCw4wQC8
1JS5Y9BGWT8
1NIhv6fCqAU
1TTjOMYZEHQ
1WrdzNJ4A44
1bflUzy_2Y4
1dH5NbVgJHk
1hHk52k-bRY
1kfwzCmYbbk
1lwYseaL4m8
1mtGB6HR7S8
1nl2QdZY2X0
1osIAtrto2k
1qcMSTMZ6ok
1qj5tOq6EgY
1s9NcgkuhXE
1teEi1piQrM
1zxUExdIN_4
21l4imBvLyM
22Ty9QW1x5A
2A4DCpO10-A
2ASbKuqCcXY
2DgCpZOdskI
2DqwNT5qe60
2F6EkgAyuRA
2H4TxQIqYM0
2N64oGaFQCA
2PqJyqMZ0Eg
2PzOkiRDXPE
2QpIUlv0e28
2TYyRyqesyg
2TdclW7RzSY
2UCvs-Lw4rA
2UnLgXCpCEg
2WMITcgSHvM
2a6Ksm1OTEU
2f2j_OwVMKg
2jK3W_RSEwE
2kZFHoQyg-E
2oyZ7QyJTB8
2sWo_eab3Vk
2wUawmvw_0M
31jNN10-Cmk
32F2hQTncQ0
32fJ7wnRIRA
32xm_1KSWVA
36nUFQFm7HA
37ytFdg556M
3Lo6MKc0dIM
3VTNWvtkob8
3XtPggjBmA0
3Xz6w2w2vfE
3ZX33fUL6t8
3c8E-iS990Q
3cet4bO40V4
3cihIb_sBeg
3gnkVs6dFKg
3ir7xA1JlKM
3jPGl_nADAA
3ldWeYEGYdM
3nFcU-EODu4
3oJV4kGgy4Y
3sUQ_jBDJXI
3tQ0eB1Ay4g
3zbH-FB0cUI
42PTZv4WRi0
4F2dGseG6NI
4HrGpWlT9GA
4MvY8VJfH_8
4NgJbp7K6Hk
4Vc3yKgcUMQ
4X-XwLNtLgE
4Zy40_BAGdY
4cQpHG-a4SI
4hWWFqVDAIE
4rS3CWZOeE4
4s5KO1n97GU
4uQrCTkPrFU
4udV2Ai-Kgk
4yjxdO6ObKM
4zk-UQSOrWQ
50yRwPa537A
52kEHcp5PKA
52sEiZzYP0I
57ySNIcvCTg
596LMJTJFsk
5BZKrlxBREM
5HNFsjUwolI
5I-Wr_s00aU
5KWAgKR9JBE
5PZTqQrvSck
5Q-XeVxthOk
5RXRr8PKunk
5UkXnEjv7x4
5XO6-HrTajM
5ZNpKymvyo8
5hXI7YJt8o8
5kEJl7hdIaQ
5lLQtbBp6yk
5n0QmwX62QQ
5nrjL5FCg1Y
5pV9DjRBIoI
5q57HFAkC4g
5sQn3f7w6A4
5to6u28pmcM
5uke089o-mQ
5wS3qtQyue4
60kSP5JgzwM
60mQQ2eKvaE
62OTciMk_wY
66j1tTlUgLU
67QojWkQLsA
6Ac3Vujv3WQ
6BG1_vfLdug
6C8bZgOnAw8
6Exj0n0VfN0
6I5J4Mf4gMk
6M4yHhuawII
6NFT31grPf0
6PxtWgIbKi8
6R1V-Mlr-ZY
6RYLs4H1ciE
6UBz3lm6ldg
6Wj1xH8AM7Y
6YHKIaxkW_g
6cMnR4PfJfQ
6g4CSh04KQg
6nGTOTGuLOg
6nSVHzkus7U
6qLzjy7Hr9g
6t4tBmbPko8
6wFcp6cFsCg
6xvp_z8ceHk
7-cGz2eLyb4
72sL_WEQedU
74s6eiBBBFU
753t24hUTrg
79CmTSZ_g5U
7Au-MRjoHYQ
7JTCe5n2zSg
7MEdFIXTHP4
7MvpkJaCwRA
7NCUAyTjUnU
7OXb_xekQY8
7TQ36plrNjU
7XWPdWhG_jY
7Xjxdh_6adQ
7ZUp3ddSYoQ
7c7_k56DiVY
7cKGVYT3W8g
7d6Sn0EOilI
7iIyA9QWopA
7lNlME-LDZY
7rZNgb_ncNE
7u8Od1nDs48
7zB5aLrLyAE
8-Qtd6RhfVA
80TrJa4ZlrQ
81SoA_cBKMM
85_TFb48iw4
87TCC0iWB3Q
88N5kx7WjZY
8Fw70WGcK20
8JgZruMJ0fU
8L9w5vTlRRU
8Lxb917AXF4
8Rwnraf0sq8
8SBIXJSVdTQ
8VH9UINEI24
8YGfjhjcKY4
8d9jn7RNkcU
8ezyDlUF42A
8ftE8A4S6FE
8gH3F_A-974
8hTryXqp6g8
8hnvY6_Jpvg
8l4CKLqElOY
8mOjYFwmgxk
8sI5XaKs5bk
910K7B_1b6I
91ZqkjbtHFU
92gq7UYoau0
976SFoINOU0
99CLGZNuFqE
9Ax06n7V9VE
9IXcZ1G3d8I
9R5PY3IDaTU
9TdjoZKXKT4
9TmCyAjaxs8
9UVl4HQ2V1Q
9Z50YyMFCaI
9Z9eqr8sNIM
9Zr4wG3ToHY
9_8t_HSG-uE
9abusStvqCA
9acmEoE7dOk
9b1eBqIoR7I
9ccIlIId_R0
9fBLYpYwyek
9ghdculyWY0
9hIxfE8gDYk
9jkDqITkXIY
9mWX6bkgY00
9o1CIJJFV6M
9rQhskjdLXM
9uSSBu-5eSY
9zb0Oo0ryEI
ABeOnnq6k8M
AF-Qlsd2c8Y
AFxN0BNd-D8
AJ9QNdP9fHk
AMHe5wrNguU
AMYHNvBAQxI
AR_rgqo1xYg
ASQo9bIPch4
ASysfc-IYw8
AWuz9V0SH-M
A_LjUmoT8QI
A_vg1AT1A-Y
Aa3ycuHbykQ
AjDp9MClUY4
AoHvkrcr6EM
AqS3Pms8M-U
AqV5bUpbCpY
AumWr7lU0mM
AyjYSkr6sRk
Az9lWdqebaU
B1m-GQotG-g
B2-kRFave8Y
B4FcRYPlDOQ
B5h8o1HZWmM
B6GQfJmAf0A
BCSdVZtWOaM
BFUis11qaQU
BGwKsNL22NA
BI8WvLUg8K8
BIOk_BfXUzk
BOSkQ5MnjT0
BVUSo_hs1Ag
BZOiyQkMOS8
BZS0amkjOhI
BgNpATWfR5Y
BnaCRU3vjTU
BodnYcHGtiA
Br2Iqs6vWsQ
BsMVEFmwFwA
BwPhcLw5B0M
BwQsh1TDSMM
Bxo9YgIpr90
C4JRCFLKKHQ
C6w4s5gwaLI
C92dVS8NIw4
CG9y2uXXFcU
CJcfcj4C5-U
CJeKpR0oui0
CKlSFlnmDcY
CLg-WWagkKk
CM1u29BeqC0
CQVg8Tv5yQ0
CRWhQuIxspA
CWyrp3hu4KE
CXRtk0E7aN4
CYGLevDEj08
CZMj9iRTAI0
Ca9mh353EDY
CbixFksNU48
CdXWpvps_pg
CfcQdHuVZuQ
Cg2ZSaVOHsg
CpQIcK-zetg
CrjjN2sXB_g
CtTkHszSo1E
Cvoonq7Resk
CzVscrIPr2Q
D0Ob4g8fgf8
D1CUNDwCQdY
D5OPXddiYcA
D7lenSSqbP4
DA_G8iK0bAI
DCNBymqdhEY
DIIjaKwDsf4
DLux6jxNrcY
DPxumNR7UtM
DQYBVdPPDsQ
DWA_cVm7mEM
D_6XgzowY5Y
D_vcr6dHV78
Dd7vMuw3ExU
Dendav4u2yk
DgrXeBJ66_g
Dgy05UCXLQ0
Dj1b8jUzmZk
Dnu8fmx4CI8
Dt_KmEfx5D4
DtnKKb_wtxI
DyGIvs9zrVA
Dykye9vhypU
E10WcB1qFVY
E2MC3Y7hR-Q
E2SkzWC4siU
E3nTHnkZybk
E5zwwtfNomY
E9MEZXpjrYg
EK7Fq2gkFrQ
EMsmL9NfkuI
ERbI9G2jBbk
EWqs0qdXeZ4
Ec6ICpkHAR8
Ej2_mj3Fyu8
EjFrczNSRE8
Ekqq6DE8vGE
EleUObRXQ0E
EltIpB4EtYU
EpEWPFtZjUI
EtC3WoFLxhE
EtH-OOIcykk
EwQ1tPF8MgQ
F2qgSTB8Dyk
F4Lt0AjMxEY
F5zmokcqM9g
F6lzCUQd-v0
F8UGIKR3okA
FA8iUBEDIS0
FArSFWosHq8
FMG5_iBd0Sg
FQgc6e_vMVM
F_AbJs9-EU4
FePjKBoFICA
FlVnh6hO7l4
FldsntzddF8
FwpVuQi9UQQ
FylbqN6XYBA
FyzzDtzfC0k
G11lr2N7gjQ
G7SOs1s-i54
GEHrIGP9M4I
GF-YNEGTaCA
GGgGjtM5ipg
GHVtiJ9xd6A
GKSp2Ogv564
GLd_x_WFaqk
GO40_Zi9WAQ
GYpU99BTbzg
Gazuoyf-lm4
GcztVMg3Z-k
GfL3GCFU-b0
Gfr364SyQcg
GgkBFkVas00
Gl9EsvAXgDY
GqoUzm8ej5k
GqyS_L71nEY
GvTta5ib0MY
GvbSiwZYh6M
GwEucKnTerM
GwQaekb5onQ
Gx6t_x0hF7U
H2IqRgAHpFk
H5LuUPA8Hvk
HB3m-Ni8jSM
HB5O-aqR3G8
HCeaUF9sCEE
HFT_bGnNLK4
HIz9E3qAfiU
HKP1VjQg6Jg
HL6w_Xw5C_g
HMLCN3DI7d8
HOwqo56BKQU
HQ0mIz0I14E
HUd9ucjaCLI
HZLtwCKwYYg
He5HE4y4vmE
Hrdu4Dh6DoU
HvcQdJcN6mk
HwFvwhwuJ1E
HyiSxfdytoA
I13xK_g27rk
I1WOEwRVjdg
I1hZSLXAVn0
I6G6NPWe2mU
IKCDefMkH58
IMK_XBI8Yx4
IMLVbEzM0DI
IMPzAdVOhKY
INC6bICXmKI
INpDGmujLSQ
IO4Gyto7KrM
IO6XF9RPjC4
IRV2wWm6BQk
IVH_6ciQfZk
IWPE3Cdnc34
IY699mzi7K4
IZZaeo4jBBM
Ibr429pU7og
Ig8JNw-Znt0
IoBDTd9vHJE
IpPHCLrEmSI
ItjPn8WxVSA
Iu9GPd4A-nk
J0Fd5vJGAWU
J0R1p5suhtQ
J89Qw3MOIo8
J97G6BeYW0I
JASPk6dLvKY
JAjZv41iUJU
JAs2K6mQSmM
JBjEKPOKlVE
JC7AiQmAynE
JCQk7kVcvUg
JDSrcVa4aS0
JEZvmDtj4UI
JJoyWK58CvY
JKMuJy12K-U
JMknA_CCnW4
JNKaTZIK3NM
JPJ4l7QZ9eM
JSFiOF7xGfE
JTtSyFi2zB4
JVUUCE7aluM
JW3JyccEXRI
JWgNUPI-dMI
JXygbkEjNIE
JbDvSPJofeg
JevaHIjSK_g
Jl77C9JQa1I
JloyNp6P7oQ
JmHkHHkiGwk
Jn0umG_ZuII
Jn8Yuo20G78
JoJ13RWsLV8
JqJaVvLO748
JrhO9vyrztQ
K1Zg04lJXgM
K2ICK70lEr8
K2QqsZAoQwA
K4SOS9okas4
K4UhxeuSKw4
KDP239KYaPk
KFMs2HpFoWE
KFgr-elM9Hk
KJqRSDyfFvU
KLMnpgVBpJg
KM82rY3WEKM
KP6Zm9vl3FM
KPjvj0P-vAQ
KQVHtp609p8
KSMA1gYL4ZY
KW8HnpSp6wk
KYyMI0gyLYk
KfYBh97HU4M
KgL1jozNYGU
Kih-xzajeGI
KiouveG278Y
Kn3lgskKA08
KnArDKQqmj0
KtT7zvYJekA
KxQTLP2KeaE
KyPbLw3t_RM
L-a-dKhuYj8
L2ZNvl1saLY
L3r_pfmelVQ
L77b57erQ4M
L8lBrTXUVSg
LBpwiaWZJLE
LCdTfQXDK34
LHCJ-Xq-Gak
LIuep2DazL4
LJb_bqfzuxE
LJvQxGd6uHM
LLU3zNPpvEg
LSUUsAcUSfE
LSvWz4Iul2Y
LX9iblqBfXA
L_gPzOxZ3TE
LadqiXXBnl0
Lfahe1cmO6Y
LniJbcPpxLM
LqSBrVbVBVQ
LrtiKvSd2r8
LruiDyDmzP0
LvaVsnHbIbI
LwLWo6kUst4
LxpbOl5TipY
LxsUZUI0MkI
M1bQ-gKzd1Q
M1dA33-Jvdk
M4BOOuJ2oiI
M8BWBrhaeKI
MG-YQAWJeFk
MJMG4-VH_rI
MQ1dhCmTpVg
MRJlb24X7X4
MUw11H-aCMY
MYQzJKeYP_M
Md52nIb2eb4
Md7PssTAvz8
MdM3I-iqyq8
Mdo5wdQNa_g
MmBmhxyb0t8
MmQzk3f1fKQ
Mp1ADUs1vNM
Mrzn1MP2Mb8
MsGmoVknaZQ
MwLl9q9VEu8
N-KZR7rxSq8
N0fiGtDcKPU
N2CeF78AGzU
N3kL6SKSoYY
N6oO9n6t8s8
N97-kDLucyU
NBnnwiyuI4I
NI79s7lcW-4
NLCaT3c7J3I
NMvTYhPnL8A
NQXGXfpV9d0
NRH-PJoaP58
NYm5sB6rT_E
N__Fekod1fY
Na-HWnYIy4M
NgX3JS-HnB4
Nnb4jrQbHyg
NqVXPZs66k8
Nr_arsIE7vA
NuzV3h2MyN4
Nvd5uk5z24A
NvghNK7TRjY
NwBzGfKY_gE
O2nAICrb7Ck
O5hUR5wPFHA
O5o5EC7fOaA
ODM4I95VxME
OHn2YUQJ98Q
OIM_5PBEU-E
OLVAUYDNbpk
ONZUfwAdHKE
OOBBD_5_KaY
OOI-raXimT4
OZ27w3gWDN0
O_kXOnrYYRA
OdHj5qimfl0
OgpVYTnsmuk
Oh_mPlJybbk
OlIVt-Yv6I4
Ov65fQWy5PQ
OvyUoZ9K8R8
OzK6zn-falA
P1DJRdHUDw0
PAab0M_6fuQ
PBB4SHQHTbY
PBlRQTRAc-Q
PEF5rIneraQ
PFtxU_0TNpQ
PGWDP-846Rg
PHGmEK7_49M
PLoNvGq1by4
PMEZ7bRnHwg
POyEK_C4XP8
PPpLxfSvT9Q
PTEfGjlVp2E
PbpuctabwvM
Pdec8clsV7Q
PfX-aCuxfBk
Pl28IILPDTU
PpC1cE7ZeEQ
PqgKFsK7f-Q
Pr_-cUA_PY8
Py9b7iSl5w4
Q-n-BRvSaK0
Q1aHsYftvLI
Q2AleRgc_rQ
Q4p6Suko25A
Q86xxuh2V2g
Q95gTsdk294
QAT_a8Mhyu0
QE0sj2hwpYg
QHIxH3wbc84
QHe9ReWwkhk
QJmBbhjFP6Y
QKdOYRneU44
QMH4dkyDL90
QPj6e1aqiB0
QQy4vasH4B4
QStFQDB4Dt4
QZu4UYrqoUE
QfrwYsrqo8I
QjMIA-vaLaQ
QmH-cLBdsRE
QpKBRC_RJFM
QqQ4KZ3oT74
QuSQg5ZVbp4
Qx4G7pcx0yk
QzhdLt9e2Ck
R0lq_XM4NHk
R3KuHxrtrg0
R5k57QaVB-0
R8iGRccartA
RAljwauggrE
RHZoQcQozYM
RIOPPpmIEo0
RMs7NvrQx8Y
ROgY6F0oRq0
RQJerxixvEk
R_gp8I-ZyZQ
RaNNNrmcJgY
Rd8ohUO2-4M
Rem9ZfiJ8wU
RsytUXscl4A
RtwI2JocwO8
RycF0ub2Al0
S-3nfpc_GiQ
S09YWIykt2Y
S1pznvqNV1U
S5ybbawRqbg
S8__zp7PdZM
S8ncSP8ozvg
S9Yw8WoTT-U
SA7yCKxzzr0
SG7NjIq6TV4
SHmjM_EHrzM
S_nJNks8VTI
SbwAB3vscnA
ScjyYj7UdsM
Se5zpcXtefY
Sev0xmmU6LI
St9r4pc2QmE
SxLTYDKWSrg
T207B-r6vYM
T2wKkoSDqWE
T5sqFzCvswg
T6lKxkzSnSY
T7J-yADVRY0
T8rCQh9t17k
TAwycgsaMlQ
TKnlUbM37yg
TQhtAjJikUY
TRUhLSURFO0
TSf3bS6ry-8
Ta2JyiazUtg
TiwW0AMIRiM
TjxVyLTu5vI
Tljta64txY4
Ts2qo44aNi4
Tt3J0flYZt8
Tuybdi-iVF8
U39pKokdcsE
U66C5FlolYU
UD70qaaI4E8
UEaUs3YXlt4
UGvSXyNJAAE
UHt5LXLncvE
UOvWr-vMO94
UPhGGR_Xyo4
UZ9JbmQiqYM
Ub3iwsiI7GM
Uk6U9ROn_MU
UphNkxbNC-U
Utskmy7lEu8
UxugWPhL_mU
Uxw75plvyx0
Uzj1vex_H24
V3D4Uz_rH5U
V6jFEq1T-wQ
V9FDxlZ_sik
VJgnv314808
VJkYLU2seF0
VNqmDmYMilU
VStR7iOeR0E
VTvcSqAVX5Y
VUqG6Z1wx9U
VXAFdiRubCI
VYLpIuUV-LQ
VZBGZ7YT-vY
V_DbpHyyzng
VbwRYFMPZS4
Vc_xioDS1DQ
VcltNvduSX0
VguGd-Tav3U
VhvGCUbYLrE
ViMyWYbZmYU
Viep2gZx4qg
Vlg7J0Y1RfU
VpobvFPR6hQ
VsbCOtZYGzY
VslZoZD9njE
Vtb4xCFygiU
VwWGCukz7O4
VwbCKToosT0
Vzvf7SiOk78
W0KPwTy0W9k
W1_pSpqCS58
W2Hz-M917-4
W3xE6BRra2g
W7Fk8O88cuM
WBjchfwi1jA
WEM-Nc_dn-Q
WEmzGVoYHO8
WF-I9BA5F00
WIpxrtf7QbY
WJbZg1PgzO4
WM3v85H1jX0
WOPcKqhV-o4
WPgsgZ58jW4
WQtwfwmYpsY
W_eG_bDhD8g
WcfONpA14iU
WjyQTmJBXFg
Wmtf2SPr3-g
Wqwlx0LrsVA
WsrS9XszuN0
Wtof98E8cC4
X0LUf1PpuPg
X0s_uUogolQ
X2rOWkOcq9I
X87NbBTXixE
X8G1YkurALI
XFBYZK_JCiM
XN3grZJUjUY
XOHO-j4ay6g
XR7Cy9hoBPA
XTvwESqfIAE
XUIktTOH2BY
XVvHbi0bZPM
XWhQlwaKY7Q
X_aNjxWOocA
X_yF9QFErpA
Xe_LsrAKP6U
XfSWO8a1PlQ
XfyGv-xwjlI
Xh6Vs-NiVTI
Xhd368LZRKA
XiQdXbkGwkI
Xz1LmdofZoQ
Y3iFuNgJbcg
Y5AL8FruOJE
Y98ry3671Rg
Y9tD5qMeDnI
YA91m_--WAg
YCsyTd29i9Y
YEbm6yp6Wf0
YEvH2mPyxWI
YICvHsKmjzY
YR3rrZV8HVQ
YRmJqwW36nA
YXhzyf02peg
YYAr1h1Rxq4
YdkyfGBc9YI
Ydzo0ypudQM
Ye9SRh59pjg
YeDDPumPYNk
YeJI1jeg1n8
Yhn2qguQfUM
YiIPbFuuboY
YiQfe-oz2C4
Ymtiig6l3zA
YqYQgtpKXrM
YxTpeB3vb9g
Yz19HLEEXqg
Z-cxMdEvsZM
ZA8GzhFh_CQ
ZCpSiSS0Yv4
ZFlgzUwX2UY
ZFwcGVGNOzY
ZHVkA15HmbM
ZMAhHaLg0lQ
ZMWGfX2jBDA
ZTReTNm8abU
ZTqXRVGb0dk
ZVxbRnbFRB4
ZWrybTGDFHY
ZfSlQJk59Ng
ZiITCnnFbwQ
Zo9r9FRNILI
ZqVpqelyLFE
ZrkB236xr94
ZtVoHOayVmg
ZwQN4F8bLyk
ZwZigB_LmTg
_2m2OJgYxgI
_8VbgtVMf8Y
_9hPFlSnnNk
_A2Vo9kWdJI
_Byv0Zx9fHU
_C0i8IuHXEI
_CKuBMAlO2Y
_DGyROs4QD0
_GwKaHwd7CU
_JjAi11j_a4
_KatXIHOd4Q
_KndwmMEwXA
_N6AAre8Zag
_PQPw4bLrlE
_QPz0ui-7uM
_RALKmqnsYA
_ZUj2aYMDUA
_aLLrQVX3L8
_awmMMKmoS4
_bQi08rSz1g
_ex2PNXdhgM
_f-KFSSzzlk
_f9oZnhgp3U
_fCqEIn7ccA
_t7J9U_NFvE
_uXvihpw_KM
_utjrrpM4lk
_ygn-XrT150
_z3a81ge5AU
d5kVw6fx-n0
d6H_jPqjLKk
d8PmuZLlBqY
dC7y9qB3tcA
dCZ6d1fElpU
dEJ_r_03A68
dFtBokeHkWA
dH9Y_rby-jQ
dIqGm-Ib1oQ
dJ2F0RhvIhI
dJv7fFtPKwg
dK3Iyl5_n-Q
dKYxRDO2G_0
dL892eus81M
dNtwaasl2y0
dPKvHrD1eS4
dPyDyldwUlE
dQXKNXz_ZCg
dQr9asiswzA
dVxkV5KhTV4
dW-iPDFJtBU
daUGeFjO8NM
daVDStDLkDI
dafRjRtvfck
db2XgUgN94I
dhGvYjHzcIU
dhPZNnhZ5vU
dhrERViJSm8
djhv9FDyyIQ
dn0W0KJ4g48
dnRcZ5RMcbU
drAenNuBpPk
drXGjh3jRH0
du9qrh2DMck
duDHe8TGplk
dz6SNdKBnHI
f2HnkDGuBqk
f6txLUgJD54
fDjWczxy374
fF7dCRNQiek
fFRuvcYyHo8
fHEVfBzEANI
fHui_Q6ld10
fSYFsrurYKI
fTA5HOa87pM
f_O46jTHCtQ
f_bFPNbfNhQ
feNKdaMb1YU
fihS3YUlolY
fn2i9Z9TH7U
fojJ_H7hxkY
fsGqVYlUtbY
fw2vF05J4Z8
fxLagrFbPD0
fyOFbCmTzaQ
g-9YuxM5aNQ
g55rOPM-lWI
g56j1HhU0RI
g6J05QHC26Y
g7JLG36ZO-U
gANsmfNjEnA
gAuNCvWbRpQ
gC2cMr2fLgU
gFQF8i1Wmf4
gIZT0Ew0ugU
gJODa0YqzKg
gJfrTzkB4KI
gKd6_EqFI20
gMHbBr-dHJA
gMHiLBFNcNQ
gV326cyB0lg
gVro-wGpMpY
gciQEPUSnvI
geO2L3fzF5g
gfoDXFrih3A
gzwtHkUaOfk
i44Av6afijM
i5VhqPTHmv4
i8LUX2JEzyE
iAGkZmwB9XM
iIYqxBUM5dg
iJjhc5Lf744
iKKF5yuxvg8
iLllLj9n_Ng
iMzqR-XzueA
iN6u1fuJor0
iQEpDszK0Ck
iRgKZ-kHgOM
iSVEAqBVCwA
iS_S6ByJYLI
iUofnjY-HXI
iaqGLfsucJ8
icF2xJsSyX4
ie-oS8vQaXU
ifMbcPmdH6U
ik--FDrR-jM
ikGl7DPXUK0
imQ5qMqdoJM
is9Q5ZmuhqY
iyxhUqxusuY
j0QmUup77i0
j9oeKa-Brv4
jGpH3AD_Mj8
jHZt0naRnbM
jIZjtqUmSEM
jK-63yMLNk4
jOICrFbQGI8
jTPypnbFNFg
jUfGVwaB5UM
jWxMP8eK_a0
jYLh7_X2Cik
ja8T_rzO7F4
jiLKEjSi8cQ
jwD7bJrYH6U
jy0oXpqobwI
k1XG91UBQa8
k20meh6MRKU
k22zlWRzbk0
k42Ne6jiEIQ
kI6FITZycek
kIleZ3gCF3I
kK6mIPzBWWM
kLHRQJHFAqI
kMOQaxf9pag
kMaSLzivnWU
kMyx68jVRB0
kO2rU72dh1U
kOW2UfaVNzI
kOnN-Zh6HHE
kT8lFdOxVeE
kUaJ3OScyGY
kV4Y5itBXJI
kXP0THL-oO8
kYph5ulP2qA
kdbcA0K--d8
kefdajmcDzM
kfLEIN0X1Y8
khvpSnWYFJI
kiPd0ezjme4
kjMUz28ZItM
knXSucCKnmM
kp1hPho9RKA
krEeX5G9AYM
kzTG83mLRKU
lEutFrar1dI
lNAAzs4C-lQ
lOeQLJCVOdM
lPTlx2dcoNU
lP_kU6akp_A
lPvB3FqqigA
lVuuisHvptQ
lWQeanGvLUE
lZ2V_VfswRE
lZ_FN1Rk7zI
lchvWAnJSwY
lfdc3kG8354
lmq7wycWwu8
loUuq7rPTH0
lpHJP-oaLvI
lq4S9L1qjTg
lqLnrqV6I5I
lqeGgoPttAg
lrrEhc3SRJ0
lshzaYhtOyA
lu7-Pp36zWk
lxPU9pAClAA
n4h0_9u0EOM
n7H3KfMNpX4
n8KdfomQHkQ
nAFELKYkiuU
nClrDGP65A8
nHuO80WbrDI
nIQl9KaLfSQ
nJ0YMjp-c94
nLflozGJPAU
nSGIFs-s_UU
nSuBzZR6llY
nWUMAYb31rg
nWgf-odA50Q
nX-Ikn3hK6Q
nXCv9A2kJ64
nXyIaDgTBAk
n_yAt90fzFg
nazSmyMgYnY
nbV1seoE0KM
ncpDzJH5EoM
ngH9-eyRrhY
nlvQYF62j6A
noICBDh919I
nq20aTEWppk
nqWV5s7P8QY
numUKaH34rQ
nv0PfJMa95M
nveWpFwDTnk
nwSquF9eHck
nyRrm-cCj9g
q6D0bJXBS8Q
q7KObLnrIyM
q8BOimu8oNM
qAwdZBanT8g
qAwrjX0928Y
qG3OyONVbEQ
qHTURjMqh-E
qLGqcIFu9DA
qNdoOd11Vi8
qQarsq-1ykE
qY8694ZyjbE
q_t-KXiV0PI
qcmedjz_FpI
qcw0IuDr-28
qii0YXVkFjQ
ql0lY4qKBTU
qoy7XPz9XXE
qpfJTM8JvZE
qpzS1JD1Zkg
qusGm94uFJE
qvPEkkiAoY8
qvaztKvcasU
qx31aecUA5Y
qzhAGXJFFBQ
r1gt7XUUCp0
r2IYMx8lmp8
r2tXTjb7EqU
r59S0s0S7AE
rCErhY4VKCY
rEPw0DXpY2Y
rGlwJ0B5aOk
rH_bZsO24tk
rLjIJlz6ZAM
rMMpeLLgdgY
rPmDXv-fBRQ
rQXl6XT4BCE
rR4iKEIU0vE
rUsH1dHWUI4
rW0nHlwPWu0
rXE2CiCntGE
rXqGbNinhaA
rY_siujVJEg
rZr5KTurMXk
rasZGZpQsy0
rcOTAlcM89Y
rd6T_aTbppE
re6muxqRs7Y
rf1sAHA72uY
rgQ1ffvr5W0
rh1ZZXKW5UY
rhmAyFY6dhU
riDnpJihZ7k
rkfeOy88-_0
rlveYN26dLI
rnPIDHlxiH8
rpPeJFFzJ5g
rrAEg5wSa30
ruz7xePInwU
x2wXOLHojrU
x6eE_kkxXT8
x6jQw55IeWQ
x8zGj1UM2tQ
xCW_EFv6wC4
xGnAi4F0CG8
xJBqbv8GRZ4
xJVXL5oCZCQ
xSGpX3isqUE
xchGJ4UxCPE
xeRS9N1aV84
xfSL-OdZYbk
xgy5LJAGFbg
xi9wttwNN8k
xiZ9XlqhrBI
xm2T2LXZLtU
xyQVFAmrsh0
z-iZjn2VkBU
z0Pt-x8eKkM
z3TEPkzF32M
z3zkU8zf1b0
z4GO65w1t4A
z7WZ27g_r8E
z9aeU2Qw1S4
zEO-9wibPJg
zEyifMatos8
zFuoStIj_Y4
zKjwEImcdgA
zMwhbVhbXWA
zNLpPlqyS48
zOf7NCMa4ZA
zTUZ4-lQ394
zUrnx_LQiOg
zVtMGUzr4fA
z_mb9_FHdgI
zaDMhrzC8eQ
zbf0F1X0Z24
zjygvoYw4jg
zq_v8lS8uJk
zvmOwg9fchY
'''.strip().splitlines()

# Changes:
# swapped positions of cat and dog
# changed gophers to gopher
# removed hound
# added mouse

for line in difflib.unified_diff(lines1, lines2, fromfile='file1', tofile='file2', lineterm='', n=0):
    print(line)


# In[ ]:


def compare(File1,File2):
    with open(File1,'r') as f:
        d=set(f.readlines())


    with open(File2,'r') as f:
        e=set(f.readlines())

    open('file3.txt','w').close() #Create the file

    with open('file3.txt','a') as f:
        for line in list(d-e):
           f.write(line)


# In[ ]:





# In[ ]:


from google.cloud import storage

output_list_normal = []
def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):

    storage_client = storage.Client()

    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)

    print("Blobs:")
    for blob in blobs:
      print(blob.name)
      output_list_normal.append(blob.name)


# In[ ]:





# In[ ]:


from google.cloud import storage

video_list = []
def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):

    storage_client = storage.Client()

    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)

    print("Blobs:")
    for blob in blobs:
      print(blob.name)
      video_list.append(blob.name)


# In[ ]:


if __name__ == "__main__":
    list_blobs_with_prefix(
        bucket_name="youtube_videos_data", prefix="outputOCR/", delimiter="/"
    )


# In[ ]:





# In[ ]:


# this is a video that I have uploaded on Cloud Storage and I obtained the URI from google console 

for video in video_list[1009:]:
  print(video_list.index(video))
  gs_URI = 'gs://youtube_videos_data/'+ video
  temp = video.split('/')
  print(temp)
  print("here", gs_URI)
  #output_uri = gs_URI + '.json'
  features = [videointelligence.Feature.SPEECH_TRANSCRIPTION, videointelligence.Feature.SHOT_CHANGE_DETECTION, videointelligence.Feature.OBJECT_TRACKING, videointelligence.Feature.TEXT_DETECTION]
  config = videointelligence.SpeechTranscriptionConfig(language_code="en-US", enable_automatic_punctuation=True, enable_word_confidence=False)
  config1 = videointelligence.ShotChangeDetectionConfig()
  config2 = videointelligence.ObjectTrackingConfig()
  config3 = videointelligence.TextDetectionConfig()
  video_context = videointelligence.VideoContext(speech_transcription_config=config, shot_change_detection_config=config1, object_tracking_config=config2,text_detection_config=config3)
  operation = video_client.annotate_video(
    request={
        "features": features,
        "input_uri": gs_URI,
        "video_context": video_context,
        "output_uri": 'gs://youtube_videos_data/output/'+ temp[1] +'.json'
    }
  )

  print("\nProcessing video for speech transcription.")

  result = operation.result(timeout=100000000)

# There is only one annotation_result since only
# one video is processed.
  annotation_results = result.annotation_results[0]
  print("Completed")


# In[ ]:


# this is a video that I have uploaded on Cloud Storage and I obtained the URI from google console 
print("out")
for video in video_list[1018:]:
  print("in")
  print(video_list.index(video))
  gs_URI = 'gs://youtube_videos_data/'+ video
  temp = video.split('/')
  print(temp)
  print("here", gs_URI)
  features = [videointelligence.Feature.TEXT_DETECTION]
  config = videointelligence.TextDetectionConfig()
  video_context = videointelligence.VideoContext(text_detection_config=config)
  operation = video_client.annotate_video(
    request={
        "features": features,
        "input_uri": gs_URI,
        "video_context": video_context,
        "output_uri": 'gs://youtube_videos_data/outputOCR/'+ temp[1] +'OCR'+'.json'
    }
  )

  print("\nProcessing video for speech transcription.")

  result = operation.result(timeout=100000000)

# There is only one annotation_result since only
# one video is processed.
  annotation_results = result.annotation_results[0]
  print("Completed")


# In[ ]:





# In[ ]:




