### PEP - 8 Import Standard ###

### Standard Configurations ###

config = {"train_epoch": 300, 'print_every_step': 4, 'num_validation_steps': 5, 'learning_rate_type': 'exponential',
          'checkpoint_every_step': 400, 'evaluate_every_step': 20, 'preprocessing': {}, 'FIRST_LAYER': {},
          'SECOND_LAYER': {}}


def update_config(args):
    """
    Update the configuration with flags.

    :param args:
    :return:
    """
    config_addition = {'learning_rate': args.learning_rate, 'decay_rate': args.decay_rate, 'model_dir': args.model_dir,
                       'fold': args.fold, 'train_data': args.train_data, 'test_data': args.test_data,
                       'cross_validation': args.cross, 'name': args.name, 'n_class': args.n_class,
                       'early_stopping_threshold': args.early_stopping_threshold, 'train_ratio': args.train_ratio,
                       'test_ratio': args.test_ratio, 'write_data': args.write_data, 'log_it': args.log_it,
                       'tasks': args.tasks, 'hidden_size': args.hidden_size, 'fund_embed_dim': args.fund_embed_dim,
                       'consecutive': args.consecutive, 'skip': args.skip, 'batch_size': args.batch_size,
                       'preprocessing': {
                           'clean': args.clean,
                           'augment': args.augment,
                           'replace': args.replace,
                           'formal_filter': args.formal_filter,
                           'remove_stop': args.remove_stop
                       }, 'FIRST_LAYER': {
            "glove": args.glove,
            'layer_1': args.layer_1,
            'layer_1_include': args.layer_1_include,
            'peephole_1': args.peephole_1,
            'peephole_2': args.peephole_2,
            'pos_include': args.pos_include,
            'kernel_size': args.kernel_size,
            'pos_dimensions': args.pos_dimensions,
            'embed_dimensions': args.embed_dimensions,
        }, 'SECOND_LAYER': {"elmo": args.elmo,
                            'attention': args.attention,
                            'attention_vectors': args.attention_vectors,
                            'mask': args.mask,
                            'layer_2': args.layer_2,
                            'peephole_3': args.peephole_3,
                            'peephole_4': args.peephole_4,
                            'pool_mean': args.pool_mean,
                            'attention_prob': args.attention_prob,
                            'num_attention': args.num_attention,
                            'human_attention': args.human_attention,
                            'pos_weight': args.pos_weight,
                            'first_cause': args.first_cause,
                            'second_cause': args.second_cause
                            }}

    config.update(config_addition)
    return config
