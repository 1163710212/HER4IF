import argparse

if __name__ == '__main__':
    # initial args
    init_parser = argparse.ArgumentParser()
    init_parser.add_argument('--env_class', type=str, required=False, help='Environment class.', default='KREnvironment_WholeSession_GPU')
    # init_parser.add_argument('--policy_class', type=str, required=True, help='Policy class')
    # init_parser.add_argument('--critic_class', type=str, required=True, help='Critic class')
    # init_parser.add_argument('--agent_class', type=str, required=True, help='Learning agent class')
    # init_parser.add_argument('--buffer_class', type=str, required=True, help='Buffer class.')

    initial_args, _ = init_parser.parse_known_args()
    temp = input("class: ")
    eval(temp)().say()
    envClass = eval('{0}.{0}'.format(initial_args.env_class))
    print(envClass)
    print(initial_args)