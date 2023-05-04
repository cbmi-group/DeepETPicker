from model.residual_unet_att import ResidualUNet3D

def get_model(args):

    if args.network == 'ResUNet':
        model = ResidualUNet3D(f_maps=args.f_maps, out_channels=args.num_classes,
                               args=args, in_channels=args.in_channels, use_att=args.use_att,
                               use_paf=args.use_paf, use_uncert=args.use_uncert)

    return model
