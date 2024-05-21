from models.proteinf3s_func import *










def build_model(cfg):
    if cfg.model == 'base':
        model = ProteinF3S_Base_Func(cfg)
    elif cfg.model == 'seq_struct_cat':
        model = ProteinF3S_SeqStruct_CAT_Func(cfg)
    elif cfg.model == 'seq_surf_cat':
        model = ProteinF3S_SeqSurf_CAT_Func(cfg)
    elif cfg.model == 'surf_struct_cat':
        model = ProteinF3S_SurfStruct_CAT_Func(cfg)
    elif cfg.model == 'surf2struct_cas':
        model = ProteinF3S_Surf2Struct_CAS_Func(cfg)
    elif cfg.model == 'surf2struct_msf':
        model = ProteinF3S_Surf2Struct_MSF_Func(cfg)
    elif cfg.model == 'f3s':
        model = ProteinF3S(cfg)


    # elif cfg.model == 'seq_struct_msf':
    #     model = ProteinF3S_SeqStruct_MSF_Func(cfg)




    return model



















