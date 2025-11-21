import torch
checkpoint_path = "/Users/kivancserefoglu/Desktop/IIT/CS584_MachineLearning/Project/Repository/S2IP-LLM/Long-term_Forecasting/checkpoints/long_term_forecast_SPY_512_96_S2IPLLM_Financial_ftM_sl512_ll0_pl96_dm768_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_FinS2IP-LLM_SPY_0/checkpoint.pth"
ckpt = torch.load(checkpoint_path, map_location="cpu")
print(type(ckpt))
for k in ckpt.keys():
    v = ckpt[k]
    if isinstance(v, dict):
        print(f"{k}: dict[{len(v)}]")
    elif hasattr(v, 'shape'):
        print(f"{k}: tensor{tuple(v.shape)}")
    else:
        print(f"{k}: {type(v)}")
state_dict = ckpt.get("model") or ckpt
print("\nState dict keys (first 20):")
for i, key in enumerate(state_dict.keys()):
    print(key)
    if i >= 19:
        break
print(f"\nTotal parameters stored: {len(state_dict)}")