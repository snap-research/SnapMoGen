import warnings
warnings.filterwarnings("ignore")

import os
from os.path import join as pjoin

import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from model.vq.rvq_model import HRVQVAE
from model.transformer.transformer import MoMaskPlus
from model.cnn_networks import GlobalRegressor
from config.load_config import load_config

from utils.fixseeds import fixseed
from utils import bvh_io
from utils.motion_process_bvh import process_bvh_motion, recover_bvh_from_rot
from utils.utils import plot_3d_motion
from utils.paramUtil import kinematic_chain
from common.skeleton import Skeleton
import collections
from common.animation import Animation
from einops import rearrange, repeat
from rest_pose_retarget import RestPoseRetargeter

import numpy as np

def inv_transform(data):
    if isinstance(data, np.ndarray):
        return data * std[:data.shape[-1]] + mean[:data.shape[-1]]
    elif isinstance(data, torch.Tensor):
        return data * torch.from_numpy(std[:data.shape[-1]]).float().to(
            data.device
        ) + torch.from_numpy(mean[:data.shape[-1]]).float().to(data.device)
    else:
        raise TypeError("Expected data to be either np.ndarray or torch.Tensor")


def forward_kinematic_func(data):
    motions = inv_transform(data)
    b, l, _ = data.shape
    # print(data.shape)
    global_quats, local_quats, r_pos = recover_bvh_from_rot(motions, cfg.data.joint_num, skeleton, keep_shape=False)
    _, global_pos = skeleton.fk_local_quat(local_quats, r_pos)
    global_pos = rearrange(global_pos, '(b l) j d -> b l j d', b = b)
    local_quats = rearrange(local_quats, '(b l) j d -> b l j d', b = b)
    r_pos = rearrange(r_pos, '(b l) d -> b l d', b = b)
    return global_pos, local_quats, r_pos


def load_vq_model(vq_cfg, device):

    vq_model = HRVQVAE(vq_cfg,
            vq_cfg.data.dim_pose,
            vq_cfg.model.down_t,
            vq_cfg.model.stride_t,
            vq_cfg.model.width,
            vq_cfg.model.depth,
            vq_cfg.model.dilation_growth_rate,
            vq_cfg.model.vq_act,
            vq_cfg.model.use_attn,
            vq_cfg.model.vq_norm)

    ckpt = torch.load(pjoin(vq_cfg.exp.root_ckpt_dir, vq_cfg.data.name, 'vq', vq_cfg.exp.name, 'model',mask_trans_cfg.vq_ckpt),
                            map_location=device, weights_only=True)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'model'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_cfg.exp.name} from epoch {ckpt["ep"]}')
    vq_model.to(device)
    vq_model.eval()
    return vq_model


def load_trans_model(t2m_cfg, which_model, device):
    t2m_transformer = MoMaskPlus(
        code_dim=t2m_cfg.vq.code_dim,
        latent_dim=t2m_cfg.model.latent_dim,
        ff_size=t2m_cfg.model.ff_size,
        num_layers=t2m_cfg.model.n_layers,
        num_heads=t2m_cfg.model.n_heads,
        dropout=t2m_cfg.model.dropout,
        text_dim=t2m_cfg.text_embedder.dim_embed,
        cond_drop_prob=t2m_cfg.training.cond_drop_prob,
        device=device,
        cfg=t2m_cfg,
        full_length=t2m_cfg.data.max_motion_length//4,
        scales=vq_cfg.quantizer.scales
    )
    ckpt = torch.load(pjoin(t2m_cfg.exp.root_ckpt_dir, t2m_cfg.data.name, "momask_plus", t2m_cfg.exp.name, 'model', which_model),
                      map_location=cfg.device, weights_only=True)
    if isinstance(ckpt["t2m_transformer"], collections.OrderedDict):
        t2m_transformer.load_state_dict(ckpt["t2m_transformer"])
    else:
        t2m_transformer.load_state_dict(ckpt["t2m_transformer"].state_dict())
    t2m_transformer.to(device)
    t2m_transformer.eval()
    print(f'Loading Mask Transformer {t2m_cfg.exp.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer


def load_gmr_model(device):
    gmr_cfg = load_config(pjoin("checkpoint_dir/snapmogen/gmr", "gmr_d292", 'gmr.yaml'))
    gmr_cfg.exp.checkpoint_dir = pjoin(gmr_cfg.exp.root_ckpt_dir, gmr_cfg.data.name, 'gmr', gmr_cfg.exp.name)
    gmr_cfg.exp.model_dir = pjoin(gmr_cfg.exp.checkpoint_dir, 'model')
    regressor = GlobalRegressor(dim_in=gmr_cfg.data.dim_pose-2, dim_latent=512, dim_out=2)
    ckpt = torch.load(pjoin(gmr_cfg.exp.model_dir, 'best.tar'), map_location=device)
    regressor.load_state_dict(ckpt['regressor'])
    regressor.eval()
    regressor.to(device)
    return regressor


if __name__ == '__main__':

    cfg = load_config("./config/eval_momaskplus.yaml")
    fixseed(cfg.seed)
    retargeter = RestPoseRetargeter()

    if cfg.device != 'cpu':
        torch.cuda.set_device(cfg.device)
    device = torch.device(cfg.device)
    torch.autograd.set_detect_anomaly(True)

    cfg.checkpoint_dir = pjoin(cfg.root_ckpt_dir, cfg.data.name, 'momask_plus', cfg.mask_trans_name)
    cfg.model_dir = pjoin(cfg.checkpoint_dir, 'model')
    cfg.gen_dir = pjoin(cfg.checkpoint_dir, 'gen', cfg.ext)
    meta_dir = pjoin(cfg.data.root_dir, 'meta_data')

    os.makedirs(cfg.gen_dir, exist_ok=True)
    os.makedirs(pjoin(cfg.gen_dir, 'bvh'), exist_ok=True)
    os.makedirs(pjoin(cfg.gen_dir, 'mp4'), exist_ok=True)


    mask_trans_cfg = load_config(pjoin(cfg.root_ckpt_dir, cfg.data.name, 'momask_plus', cfg.mask_trans_name, 'train_momaskplus.yaml'))

    vq_cfg = load_config(pjoin(cfg.root_ckpt_dir, cfg.data.name, 'vq', mask_trans_cfg.vq_name, 'residual_vqvae.yaml'))
    mask_trans_cfg.vq = vq_cfg.quantizer
    # res_trans_cfg.vq = vq_cfg.quantizer

    vq_model = load_vq_model(vq_cfg, device)
    gmr_model = load_gmr_model(device)

    cfg.data.feat_dir = pjoin(cfg.data.root_dir, 'renamed_feats')
    meta_dir = pjoin(cfg.data.root_dir, 'meta_data')
    data_split_dir = pjoin(cfg.data.root_dir, 'data_split_info1')
    all_caption_path = pjoin(cfg.data.root_dir, 'all_caption_clean.json')

    test_mid_split_file = pjoin(data_split_dir, 'test_fnames.txt')
    test_cid_split_file = pjoin(data_split_dir, 'test_ids.txt')

    mean = np.load(pjoin(meta_dir, 'mean.npy'))
    std = np.load(pjoin(meta_dir, 'std.npy'))

    
    template_anim = bvh_io.load(pjoin(cfg.data.root_dir, 'renamed_bvhs', 'm_ep2_00086.bvh'))
    skeleton = Skeleton(template_anim.offsets, template_anim.parents, device=device)

    t2m_transformer = load_trans_model(mask_trans_cfg, cfg.which_epoch, device)

    num_results = 0

    f = open(pjoin(cfg.gen_dir, 'text_descriptions.txt'), 'a+')
    animate_gt = False

    texts = [
        "A person is practicing martial arts in slow motion.",
        "A person is walking like they’re stuck in molasses.",
        "A person is pretending to be a bird taking flight.",
        "A person is jumping like a frog.",
        "A person is walking confidently.",
        "A person is swinging a sword.",
        "A person is tiptoeing across a creaky floor.",
        "A person is dancing to hip hop music.",
        "A person is slipping on ice.",
        "A person is dodging punches in a fight.",
        "A person is sneaking through a dark alley.",
        "A person is performing a dramatic bow.",
        "A person is startled and jumps back.",
        "A person is celebrating with a joyful dance.",
        "A person walks like a robot.",
        "A person is pretending to swim on land.",
        "A person is trying to stay balanced on a narrow beam.",
        "A person is jumping rope.",
        "A person is saluting formally.",
        "A person is tiptoeing through a field of flowers.",
        "A person is cautiously balancing on slippery ice.",
        "A person is cautiously tiptoeing across a creaky wooden floor.",
        "A person is spinning in place like a figure skater.",
        "A person is stomping angrily while shaking their fist.",
        "A person is playfully chasing after a fluttering butterfly.",
        "A person is pretending to row a boat with slow, rhythmic strokes.",
        "A person is stretching their arms upward while yawning.",
        "A person is performing a lively salsa dance step.",
        "A person is walking backward with exaggerated caution.",
        "A person walks confidently on runway like a fashion model",
        "A person is walking proudly like a runway model.",
        "A person is wading through water, lifting knees high with each step.",
        "A person is reacting with surprise, stepping back quickly and raising their hands.",
        "A person is cautiously opening a creaky door, peeking inside.",
        "A person is leaning forward, trying to catch their breath after running.",
        "A person is joyfully jumping rope with energetic swings.",
        "A person is mimicking a slow, exaggerated zombie walk.",
        "A person is reaching out to catch a falling object with quick reflexes.",
        "A person is carefully balancing on one foot on a narrow beam.",
        "A person is stretching their arms up and yawning.",
        "A person is pretending to row a boat with energetic strokes.",
        "A person is confidently walking with hands on hips.",
        "A person is nervously checking their watch repeatedly.",
        "A person is pretending to row a small boat gently.",
        "A person is playfully sneaking behind someone.",
        "A person is stretching one leg forward in a warm-up pose.",
        "A person is excitedly spinning around with arms outstretched.",

    ]
    
    rewrite_texts = [
        "The person stands tall, then flows through a sequence of slow, deliberate martial arts moves. Each punch, kick, and block is executed with precise control and fluidity. Their torso twists gracefully, limbs extend smoothly, and breathing is deep and measured. The pace is unhurried but powerful, emphasizing tension and control throughout the motions.",
        "The person moves forward with exaggerated heaviness, each step slow and effortful as if trudging through thick molasses. Their feet drag slightly on the ground, knees lift slowly, and the torso leans forward with weight. Arms swing sluggishly by their sides, and their face shows strain as they push through the sticky, resistant environment.",
        "The person crouches low with knees bent and arms extended sideways like wings. They begin with small hops, gradually increasing height and breadth of their arm flaps. Their torso leans forward as they simulate taking off, rising onto the balls of their feet and stretching limbs outward. Movements are fluid and soaring, embodying the effort and grace of flight.",
        "The person crouches deeply with knees bent and arms hanging between the legs. They push off the ground into a high, forward-directed hop, landing in a squat with hands swinging upward. They repeat this frog-like jump several times in a rhythmic, bouncy motion, using their arms to assist each leap while maintaining a playful posture.",
        "The person walks in a straight line with long, deliberate strides, chest lifted and shoulders back. Their arms swing naturally with each step, and their chin remains high. They occasionally shift their weight slightly with a swagger, pausing briefly to glance left or right, then continuing the walk with strong, poised energy.",
        "The person assumes a strong stance with legs apart and both hands gripping an imaginary sword. They lift the arms overhead and slice diagonally across the body, stepping forward to add power. After a dramatic pause, they rotate their torso and deliver a horizontal slash, ending with a defensive posture, feet grounded and gaze intense.",
        "With bent knees and raised heels, the person carefully tiptoes across an invisible floor. Each step is cautious and silent, arms out for balance. Occasionally, they freeze mid-step, listening for sounds, then continue with even more care. Their body leans slightly forward, head tilted to scan for obstacles as they proceed.",
        "The person begins bouncing rhythmically in place, then shifts into sharp, expressive movements. They perform shoulder pops, hip swings, and arm isolations. After a spin, they drop low into a squat, bounce back up with a chest pump, and finish the loop with a quick foot shuffle and dynamic pose, maintaining steady musicality throughout.",
        "The person takes a step and suddenly loses footing, sliding forward with legs split unevenly. Their arms flail outward for balance, body tilted backward. They wobble with quick foot adjustments, attempting to stabilize. After a near fall, they bend knees low and regain control, breathing out with relief and brushing off their clothes.",
        "The person adopts a defensive stance, arms up guarding the head. They rapidly lean back and duck to the side as if dodging punches. Their knees bend for agility, feet shuffle quickly to maintain position. After each dodge, they reset their stance, staying alert and ready for another incoming strike.",
        "The person crouches low, knees bent and body slightly hunched. They walk forward in careful, soft steps, lifting their feet deliberately to avoid noise. Their arms are raised for balance, occasionally outstretched to feel around. They pause often, glance nervously around, and resume with heightened caution, mimicking the careful, secretive movements of sneaking through danger.",
        "The person stands tall with legs together, then takes a step back and sweeps one arm across their chest while bending at the waist. The other arm extends outward for balance. They lower into a deep, theatrical bow, holding the position for a moment before rising slowly and turning their head slightly upward in dramatic flair.",
        "The person is standing neutrally when something suddenly startles them. They recoil with a sharp step backward, throwing their arms up in a defensive posture. Their shoulders rise and body leans away from the imagined threat. After a quick freeze, they cautiously lower their arms and glance side to side, regaining composure.",
        "The person throws their arms into the air and jumps upward with excitement. They land and break into a loose, rhythmic dance — bouncing side to side, swinging arms in wide arcs, and spinning with joy. Their face beams, and the motion finishes with a playful stomp and a raised arm as if celebrating victory.",
        "The person walks in rigid, mechanical fashion. Each leg lifts unnaturally high and plants down flat. Arms swing stiffly at 90-degree angles, pausing slightly between each step. Their torso remains upright with minimal rotation. Occasionally, they make jerky turns or freeze mid-step, mimicking the exact, unnatural cadence of a malfunctioning robot.",
        "The person lies flat on their belly and begins simulating swimming. Arms stretch forward and pull back in a breaststroke pattern while their legs kick alternately behind them. They lift their head occasionally to 'breathe,' and their whole body undulates in sync with each stroke, mimicking smooth swimming in water, despite being grounded.",
        "The person stretches both arms out for balance and walks carefully forward, placing one foot directly in front of the other. They wobble slightly, flail their arms to correct balance, and freeze mid-step occasionally to steady themselves. Their knees bend slightly, and their torso tilts forward or sideways in reactive adjustments.",
        "The person starts by bouncing lightly on the balls of their feet. Their wrists rotate in small, fast circles as if turning a jump rope. They time each small hop to match the imaginary rope’s rhythm, occasionally doing a double jump or crossing their arms for a trick. They maintain the bounce for a full loop and end with a final skip.",
        "Standing tall, the person brings their right hand to their forehead in a sharp salute, fingers straight and palm down. Their feet click together, and their body stiffens into a respectful posture. They hold the salute briefly, then lower the arm smoothly and return to a relaxed but upright stance.",
        "The person moves slowly and carefully on tiptoes, each foot lightly touching the ground. Their knees bend softly and arms float gently at their sides for balance. The torso sways slightly with each step as they navigate through an invisible field of flowers, occasionally bending down to ‘smell’ or brush past delicate blossoms.",
        "The person steps forward with extreme caution, bending knees and spreading arms wide for balance. Feet slide slightly as if on slick ice, and their torso leans forward. They shift weight slowly from one foot to the other, knees wobble a bit, and their arms make quick corrective motions to avoid falling.",
        "The person moves cautiously with feet lifted high, placing each toe gently on the ground to avoid making noise. Their knees are slightly bent for balance, arms raised slightly and fingers spread for stability. They pause often, looking around nervously, and their torso leans forward subtly with each careful step, embodying quiet stealth.",
        "The person starts by raising their arms gracefully, then pushes off the ground with one foot to spin in place. Their torso twists fluidly, and arms extend outward for balance. They complete several smooth, controlled rotations, feet gliding lightly and knees bent slightly to maintain momentum, before coming to a gentle stop.",
        "The person stomps heavily on the ground with alternating feet, each step accented by a forceful downward motion. Their fists clench and shake vigorously, arms bent at the elbows. The torso leans forward with tension, and their face expresses anger. Each stomp is accompanied by a slight body shake, emphasizing frustration.",
        "The person moves forward with light, quick steps, arms reaching out and fluttering fingers mimicking chasing a small butterfly. Their torso leans forward with playful anticipation, and their eyes follow the imagined fluttering. Occasionally, they leap or spin lightly to keep pace, ending with a delighted gesture.",
        "The person sits or stands and mimics rowing a boat, grasping an imaginary oar with both hands. They perform slow, controlled strokes, pulling back with their arms while leaning torso backward, then pushing forward while leaning torso slightly forward. Legs and feet shift subtly to stabilize each stroke in a rhythmic, flowing motion.",
        "The person stands tall and stretches both arms upward, fingers spread wide as they reach toward the ceiling. Their head tilts back slightly as they open their mouth in a big yawn. The torso extends fully, chest lifting, and they inhale deeply before slowly lowering their arms back down, body relaxing with the stretch.",
        "The person steps forward and begins a lively salsa routine with rhythmic hip and foot movements. Their arms bend at the elbows and sway expressively while feet step quickly in sync with imagined music. The torso twists fluidly, and occasional spins or pauses add flair to the energetic dance sequence.",
        "The person steps backward slowly and carefully with exaggerated caution. Each foot is placed deliberately with toes touching the ground first, then heel lowering. Their arms raise slightly for balance, and their torso leans backward while eyes remain focused forward. Small pauses punctuate the movement, as if wary of unseen obstacles.",
        "The person strides forward with tall, erect posture and a slight sway in the hips. Each step is deliberate and measured, placing one foot directly in front of the other with toes pointed outward. Arms swing gently but confidently by their sides, shoulders relaxed yet poised. The head is held high, eyes focused straight ahead, embodying grace and self-assurance throughout the smooth, flowing walk.",
        "The person walks forward with confidence, swinging their hips gracefully and keeping their shoulders back. Their steps are long and elegant, with one foot crossing slightly in front of the other. Their arms move rhythmically, and they occasionally strike a pose mid-step, like on a fashion runway.",
        "The person lifts each knee high as they step forward slowly, mimicking wading through deep water. Their arms move gently side to side for balance, and torso leans forward slightly. Feet push off the ground with exaggerated effort, and slow, heavy movements emphasize resistance from the water as they advance steadily.",
        "The person suddenly jolts backward with wide eyes and raised hands. Their torso leans back sharply, knees bend slightly for balance, and their feet shift quickly to step away from the source of surprise. Arms are lifted in a defensive posture, fingers spread wide, and the body radiates shock and sudden caution.",
        "The person approaches a creaky door and slowly reaches out, turning an imaginary doorknob with care. They lean forward, ears straining to listen, then cautiously push the door open. Their head and torso peek around the edge, eyes wide and alert. Their body remains low and tense, prepared for anything inside.",
        "The person leans forward, hands on knees, gasping for breath. Their chest rises and falls rapidly, and shoulders are hunched from exertion. Head tilts downward then slowly lifts as they recover. Small, shallow steps shift their weight as they try to regain composure after intense running.",
        "The person holds an imaginary jump rope, swinging it energetically around their body. They jump lightly with both feet together, timing each leap with the rope’s swing. Arms pump rhythmically, elbows bent, while the torso remains upright but engaged. Their expression is lively, and the motion is continuous and playful.",
        "The person shuffles forward with slow, stiff movements typical of a zombie walk. Feet drag lightly, knees slightly bent, and arms reach out with elbows locked and fingers splayed. The torso leans forward with a slight wobble, and the head tilts awkwardly to one side, emphasizing an eerie, unnatural gait.",
        "The person suddenly lunges forward with quick, sharp movements to catch an imaginary falling object. Arms extend rapidly with fingers spread wide, eyes fixed intently on the target. Their torso leans in, knees bend, and feet shift to maintain balance, capturing the urgency and reflexive nature of the catch.",
        "The person carefully steps onto a narrow imaginary beam, shifting weight onto one foot. They extend arms wide for balance and raise the other leg slowly, toes pointed. The torso remains upright but tense, and their gaze is focused straight ahead, showing concentration and poise as they maintain stability.",
        "The person stands upright and stretches both arms slowly overhead, fingers reaching toward the sky. They yawn deeply, mouth wide open, with a gentle forward lean of the torso. Arms bend slightly as they bring hands down, and the person exhales as if waking from a restful sleep.",
        "The person sits or stands and mimics rowing a boat with energetic, alternating arm strokes. Their torso twists side to side in rhythm with the movement, legs stabilize their posture, and feet shift slightly for support. Hands grip an imaginary oar firmly, and their expression shows determination and effort.",
        "The person strides forward confidently with hands firmly placed on hips. Their chest is pushed out, and their head is held high with a slight upward tilt. Legs move with strong, purposeful steps, and the torso maintains an upright, commanding posture, expressing self-assuredness.",
        "The person glances at their wrist repeatedly, raising one arm with fingers brushing an imaginary watch. Each glance is accompanied by subtle head tilts and slight furrowing of brows. Their body shifts weight from one foot to the other nervously, shoulders tense, conveying impatience and concern.",
        "The person sits or stands with relaxed posture, gently mimicking rowing motions. Arms move in smooth, slow alternating strokes, elbows bending naturally. Their torso rotates slightly with each pull, feet planted firmly, and facial expression calm and focused on the rhythm.",
        "The person moves stealthily forward with bent knees and lowered torso, hands held slightly out for balance. They take light, quick steps as if sneaking up on someone unaware. Their eyes dart side to side, and a mischievous smile plays on their lips, adding playful energy.",
        "The person stands tall and extends one leg straight forward, toes pointed and heel resting on the ground. Their upper body leans slightly forward, hands resting on the bent knee or hips for support. They hold this stretch steadily, breathing deeply and preparing for movement.",
        "The person spins around rapidly with arms extended wide, creating a fluid, joyous motion. Their feet pivot smoothly, and their torso twists fully in sync with the spin. Facial expression is radiant, with a wide smile, and their body radiates energy and happiness.",
    ]

    m_lengths = torch.randint(7, 10, (len(rewrite_texts),)) * 30

    # print(len(rewrite_texts), len(m_lengths))
    assert len(rewrite_texts) == len(m_lengths)

    m_lengths = m_lengths.to(device).long().detach()

    mids = t2m_transformer.generate(rewrite_texts, m_lengths//4, cfg.time_steps[0], cfg.cond_scales[0], 
                                    temperature=1)
    pred_motions = vq_model.forward_decoder(mids, m_lengths)
    gen_global_pos, gen_local_quat, gen_r_pos = forward_kinematic_func(pred_motions)


    for k in range(len(texts)):
        print("--->", k , "<---")
        print("user prompt: ", texts[k])
        print("gpt prompt: ", rewrite_texts[k])

        gen_anim = Animation(gen_local_quat[k, :m_lengths[k]].detach().cpu().numpy(), 
                    repeat(gen_r_pos[k, :m_lengths[k]].detach().cpu().numpy(), 'i j -> i k j', k=len(template_anim)),
                        template_anim.orients, 
                        template_anim.offsets, 
                        template_anim.parents, 
                        template_anim.names, 
                        template_anim.frametime)
        
        feats = process_bvh_motion(None, 30, 30, 0.11, shift_one_frame=True, animation=gen_anim)

        feats = (feats - mean) / std
        # print(feats.shape)
        feats = torch.from_numpy(feats).unsqueeze(0).float().to(device)
        # print(feats.shape)
        gmr_input = torch.cat([feats[..., 0:1], feats[..., 3:cfg.data.dim_pose-4]], dim=-1)
        # print(gmr_input.shape)
        gmr_output = gmr_model(gmr_input)
        rec_feats = torch.cat([feats[..., 0:1], gmr_output, feats[..., 3:]], dim=-1)
        # rec_feats = inv_transform(rec_feats)

        single_gen_global_pos, single_gen_local_quat, single_gen_r_pos = forward_kinematic_func(rec_feats)

        new_anim = Animation(single_gen_local_quat[0].detach().cpu().numpy(), 
                    repeat(single_gen_r_pos[0].detach().cpu().numpy(), 'i j -> i k j', k=len(template_anim)),
                        template_anim.orients, 
                        template_anim.offsets, 
                        template_anim.parents, 
                        template_anim.names, 
                        template_anim.frametime)
        
        single_gen_motion = single_gen_global_pos[0].detach().cpu().numpy()
        gen_bvh_path = pjoin(cfg.gen_dir, 'bvh', f"{num_results}_gen.bvh")
        bvh_io.save(gen_bvh_path, 
                    retargeter.rest_pose_retarget(new_anim, tgt_rest='A'),
                    names=new_anim.names, frametime=new_anim.frametime, order='xyz', quater=True)
    
        f.write(f"{num_results}: {texts[k]} # {rewrite_texts[k]} #{m_lengths[k]}\n")

        if cfg.gen.animate:
            real_anim_path = pjoin(cfg.gen_dir, 'mp4', f"{num_results}_gt.mp4")
            gen_anim_path = pjoin(cfg.gen_dir, 'mp4', f"{num_results}_gen.mp4")
            plot_3d_motion(gen_anim_path, kinematic_chain, single_gen_motion, title=texts[k], fps=30, radius=100)
        num_results += 1
        print("%d/%d"%(num_results, cfg.gen.num_samples))

    f.close()


