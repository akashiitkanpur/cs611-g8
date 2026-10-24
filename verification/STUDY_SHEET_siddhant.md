# Siddhant: samajhne ke liye (Thursday/Friday teach-back se pehle)

Plan v3 §8 ka rule: jo commit karega, use explain karna aana chahiye. Yeh sheet uske liye hai. Pehle khud kaagaz pe karna, phir answer dekhna.

## 1. Haath se encode karo (plan ka starting exercise)

**E2M1** (1 sign | 2 exponent | 1 mantissa), bias = 2^(2−1) − 1 = 1.
Saare positive values: 0, 0.5 (subnormal), 1, 1.5, 2, 3, 4, 6. vmax = 6.

| x | Kaise | Bits S\|EE\|M | Value |
|---|---|---|---|
| 1.25 | 1.0 aur 1.5 ke beech mein bilkul tie hai → even mantissa (0) jeetegi | 0\|01\|0 | **1.0** (error 0.25) |
| 3 | 1.5 × 2¹ → exponent field 1+1 = 2 = `10`, mantissa `1` | 0\|10\|1 | 3 |
| 6 | 1.5 × 2² → exponent field 3 = `11`, mantissa `1` | 0\|11\|1 | 6 = vmax |

**E2M3** (1 | 2 | 3), bias 1. [1,2) mein step 1/8, [2,4) mein 1/4, [4,8) mein 1/2. vmax 7.5.

| x | Kaise | Bits | Value |
|---|---|---|---|
| 1.25 | 1.010₂ × 2⁰ → exp field `01`, mantissa `010` | 0\|01\|010 | 1.25 exact |
| 3 | 1.100₂ × 2¹ → `10`, `100` | 0\|10\|100 | 3 exact |
| 6 | 1.100₂ × 2² → `11`, `100` | 0\|11\|100 | 6 exact |

Seekh: same 6 bits mein E2M3 ke paas zyada mantissa hai, toh 1.25 exact aata hai, jabki E2M1 (4-bit) mein woh 1.0 ban jata hai.

**E8M0 byte** sirf shared exponent store karta hai: byte = e + 127. Jaise e = 0 → `01111111`, e = −1 → `01111110`. Isme koi sign ya mantissa nahi hota, scale hamesha power of two hai.
Example: E2M3 block jiska max 6 hai → `ocp`: floor(log2 6) − emax = 2 − 2 = 0 → scale 1 → byte 127.

## 2. Mera kaam 5 lines mein (notes/siddhant.md mein APNE shabdon mein likhna)
1. Maine har format ka slow, scalar reference banaya. Har value exact fraction mein handle hoti hai aur asli 6-bit code mein encode/decode hoti hai.
2. Shared exponent `log2` se nahi, exact bit arithmetic se nikalta hai, kyunki floating `log2` power of two ke paas galat round ho sakta hai.
3. Harness dono implementations ke outputs bit-by-bit milata hai: random matrices, dataset C, aur jaan-boojhke banaye mushkil inputs pe.
4. Result: dataset C pe sab kuch match hota hai. Ek bug mila (`log2`, ek ulp neeche power of two), jo C pe asar nahi karta.
5. Kya isko tod sakta hai: agar convention table galat hai, toh dono implementations ek hi galti karengi. Isliye conventions lecture se confirm karne hain.

## 3. Prof ke 5 possible sawaal (model answers)
1. **"Do implementations same hain, toh yeh kya prove karta hai?"** Woh alag tarike se likhi gayi hain (exact scalar vs vectorized float), toh ek implementation ki galti dusri mein copy nahi hoti. Bug mila bhi, isse dikhta hai ki check sach mein kuch pakadta hai. Limitation: agar convention hi galat hai, toh dono ek saath galat honge.
2. **"`np.log2` galat kyun hua?"** 2⁵ se ek ulp neeche ka number float64 mein 32 ke itna paas hai ki `log2` ka result round hoke exactly 5.0 aa jata hai. Isliye floor 5 deta hai, jabki sahi answer 4 hai. `frexp` exponent seedha bits se padhta hai, isliye exact hai.
3. **"6.32 bits kyun, 6.25 kyun nahi?"** 100 = 3·32 + 4. Chaar values ke tail block ko bhi apna 8-bit scale chahiye: (600 + 4·8)/100 = 6.32.
4. **"BFP6 aur MXINT6 mein kya fark hai?"** Dono shared exponent + fixed-point mantissa hain. Block 32 aur 8-bit exponent pe dono numerically identical hain (C pe verify kiya). Hamara BFP6 16 values ke group aur 4-bit exponent use karta hai: chhote group se scale data ke zyada paas rehta hai (3.46% vs 4.05%), aur storage almost same hai (6.28 vs 6.32).
5. **"FP64 mein error kyun nikala, hardware toh FP32 use karta hai?"** Measurement ki precision ka result pe asar nahi hai: FP32 aur FP64 ke relative Y error mein 10⁻⁷ ka fark hai. Yeh hardware ka claim nahi hai, sirf measurement hai.

## 4. Limitations (report mein honestly likhna)
- Reference aur `formats.py` same convention table se bane hain. Table galat hai toh dono galat honge.
- `formats.py` v0 ek baar padha gaya tha (API ke liye), uske baad sirf black-box run kiya.
- Stress inputs humne khud banaye hain. Hidden X_test pe koi claim nahi hai.
