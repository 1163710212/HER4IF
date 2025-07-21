mkdir -p output

# RL4RS environment

mkdir -p output/Kuairand_Pure/
mkdir -p output/Kuairand_Pure/agents/
DATASET='Kuairand_Pure'
AUG_WEIGHT=0 #0.01
COEF=0.02 

output_path="output/Kuairand_Pure/"
log_name="user_KRMBUserResponse_lr0.0001_reg0_nlayer2"

# environment args
ENV_CLASS='KREnvironment_WholeSession_GPU'
# ENV_CLASS='KREnvironment_WholeSession_TemperDiscount'
MAX_STEP=20
SLATE_SIZE=10 #6
EP_BS=64 # 64
RHO=0.2
TEMPER_DISCOUNT=2.0

# policy args
POLICY_CLASS='OneStagePolicy_HRLPolicyDiscrete'
HA_VAR=0.1
HA_CLIP=1.0
# if explore the effect action set --policy_do_effect_action_explore

# critic args
CRITIC_CLASS='HVCritic'


# buffer args
BUFFER_CLASS='HRLBuffer'
BUFFER_SIZE=5000

# agent args
AGENT_CLASS='HPPO'
GAMMA=0.9
REWARD_FUNC='get_immediate_reward'
N_ITER=40000
INITEP=0.01
ELBOW=0.1
EXPLORE_RATE=1.0
BS=128
# if want to explore in train set --do_explore_in_train
#MEG='disencoder_swmlp_lwa_env0.1_hf0.1/up*01_es16'
#MEG='_envcd3-0.2_h-a0d(1+up)_l-wup_state-mlp-drop0_meanprefer_es2'
MEG='envcond3-'
REG=0.00001
for HA_VAR in 0.1
do
    for AD_BOUND in 0.3 #0.4 0.5 0.6 0.7 0.8
    do
        for INITEP in 0.01
        do
            for CRITIC_LR in 0.001
            do
                for ACTOR_LR in 0.00008 # 0.0001, 0.00001
                do
                    for SEED in 11 # 13 17 19 23
                    do
                        file_key=${AGENT_CLASS}_actor_lr_${ACTOR_LR}_${MEG}${AD_BOUND}_Aug${AUG_WEIGHT}_COEF${COEF}
                        mkdir -p ${output_path}agents/${file_key}/
                        python train_actor_critic.py\
                            --env_class ${ENV_CLASS}\
                            --policy_class ${POLICY_CLASS}\
                            --critic_class ${CRITIC_CLASS}\
                            --buffer_class ${BUFFER_CLASS}\
                            --agent_class ${AGENT_CLASS}\
                            --aug_weight ${AUG_WEIGHT}\
                            --seed ${SEED}\
                            --dataset ${DATASET}\
                            --ad_bound ${AD_BOUND}\
                            --cuda 7\
                            --max_step_per_episode ${MAX_STEP}\
                            --initial_temper ${MAX_STEP}\
                            --uirm_log_path ${output_path}env/log/${log_name}.model.log\
                            --slate_size ${SLATE_SIZE}\
                            --episode_batch_size ${EP_BS}\
                            --item_correlation ${RHO}\
                            --temper_discount ${TEMPER_DISCOUNT}\
                            --single_response\
                            --policy_action_hidden 256 64\
                            --policy_noise_var ${HA_VAR}\
                            --policy_noise_clip ${HA_CLIP}\
                            --state_user_latent_dim 16\
                            --state_item_latent_dim 16\
                            --state_transformer_enc_dim 32\
                            --state_transformer_n_head 4\
                            --state_transformer_d_forward 64\
                            --state_transformer_n_layer 3\
                            --state_dropout_rate 0\
                            --critic_hidden_dims 256 64\
                            --critic_dropout_rate 0.1\
                            --buffer_size ${BUFFER_SIZE}\
                            --gamma ${GAMMA}\
                            --reward_func ${REWARD_FUNC}\
                            --n_iter ${N_ITER}\
                            --train_every_n_step 20\
                            --initial_epsilon ${INITEP}\
                            --final_epsilon ${INITEP}\
                            --elbow_epsilon ${ELBOW}\
                            --explore_rate ${EXPLORE_RATE}\
                            --check_episode 10\
                            --save_episode 200\
                            --save_path ${output_path}agents/${file_key}/model\
                            --actor_lr ${ACTOR_LR}\
                            --actor_decay ${REG}\
                            --batch_size ${BS}\
                            --critic_lr ${CRITIC_LR}\
                            --critic_decay ${REG}\
                            --target_mitigate_coef ${COEF}\
                            > ${output_path}agents/${file_key}/log
                    done
                done
            done
        done
    done
done
