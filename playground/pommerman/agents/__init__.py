'''Entry point into the agents module set'''
from .base_agent import BaseAgent
from .random_agent import RandomAgent
from .simple_agent import SimpleAgent
from .advisor2 import Advisor2 
from .advisor2small import Advisor2small
from .deepsarsa import DeepsarsaAgent
from .dqn import DQNAgent
from .advisor1 import Advisor1 
from .advisor1small import Advisor1small
from .advisor1admiraldm import Advisor1admiraldm 
from .advisor3 import Advisor3
from .advisor4 import Advisor4
from .advisor3small import Advisor3small
from .advisor4small import Advisor4small
from .advisor1admiralae import Advisor1admiralae 
from .advisor1admiralaeteamcomp import Advisor1admiralaeteamcomp 
from .advisor1admiralaenonadaptive import Advisor1admiralaenonadaptive
from .advisor1admiralaeadaptive  import Advisor1admiralaeadaptive
from .advisor2admiralae  import Advisor2admiralae 
from .advisor2admiralaeteamcomp import Advisor2admiralaeteamcomp 
from .advisor3admiralae  import Advisor3admiralae 
from .advisor3admiralaeteamcomp import Advisor3admiralaeteamcomp 
from .advisor4admiralae  import Advisor4admiralae 
from .advisor4admiralaeteamcomp import Advisor4admiralaeteamcomp
from .advisor1admiraldmac import Advisor1admiraldmac 
from .dqfd import DQfDAgent
from .chat import CHATAgent

# Add advisors here
from .advisor_all_custom_ae import Advisor_all_custom_ae
from .advisor_all_custom_ae2 import Advisor_all_custom_ae2
from .advisor_near_bomb_ae import Advisor_near_bomb_ae
from .advisor_near_enemy_ae import Advisor_near_enemy_ae
from .advisor_near_powerup_ae import Advisor_near_powerup_ae

from .mad_dm import MADdm
from .mad_dm_no_as import MADdm_random
