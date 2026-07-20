def get_model(model_name, args):
    name = model_name.lower()
    if name == "der":
        from models.der import DER
        return DER(args)
    elif name == "afsic-ids":
        from models.afsic_ids import AFSIC_IDS
        return AFSIC_IDS(args)
    else:
        raise ValueError(
            f"Unknown model_name: '{model_name}'. Available: 'der', 'afsic-ids'."
        )
