
import math
import numpy as np

from world import World
from map import Map
from landmarks import Landmark
from robot import Robot, IdealRobot
from sensor import IdealCamera, Camera
from agent import Agent


def random_init_pattern():
    world = World(30, 0.1)

    for i in range(100):
        circling = Agent(0.2, 10.0/180*math.pi)
        r = Robot(np.array([0,0,0]).T, 
                  sensor=None, 
                  agent=circling, 
                  color="gray",
                  bias_rate_stds=None,
                  expected_stuck_time=None,
                  expected_escape_time=None,
                  expected_kidnap_time=None,
                  kidnap_range_x=None,
                  kidnap_range_y=None,
                  )
        world.append(r)
    
    world.draw()


def bias_control_pattern():
    world = World(30, 0.1)

    circling = Agent(0.2, 10.0/180*math.pi)
    nobias_robot = IdealRobot(np.array([0,0,0]).T, sensor=None, 
                              agent=circling, color="gray")
    world.append(nobias_robot)
    biased_robot = Robot(
        np.array([0,0,0]).T, 
        sensor=None, 
        agent=circling, 
        color="red", 
        noise_per_meter=0, 
        bias_rate_stds=(0.2,0.2),
        expected_stuck_time=None,
        expected_escape_time=None,
        expected_kidnap_time=None,
        kidnap_range_x=None,
        kidnap_range_y=None,
        )
    world.append(biased_robot)

    world.draw()


def stuck_pattern():
    world = World(30, 0.1)

    circling = Agent(0.2, 10.0/180*math.pi)

    for i in range(100):
        r = Robot(np.array([0,0,0]).T, 
                  sensor=None, 
                  agent=circling, 
                  color="gray",
                  noise_per_meter=0, 
                  bias_rate_stds=(0,0),
                  expected_stuck_time=60.0, 
                  expected_escape_time=60.0,
                  expected_kidnap_time=None,
                  kidnap_range_x=None,
                  kidnap_range_y=None,
                  )
        world.append(r)

    r = IdealRobot(np.array([0,0,0]).T, 
                   sensor=None, agent=circling, 
                   color="red")
    world.append(r)

    world.draw()


def kidnap_pattern():
    world = World(30, 0.1)

    circling = Agent(0.2, 10.0/180*math.pi)

    for i in range(1):
        r = Robot(np.array([0,0,0]).T, 
                  sensor=None, 
                  agent=circling, 
                  color="gray",
                  noise_per_meter=0, 
                  bias_rate_stds=(0,0), 
                  expected_kidnap_time=5,
                  )
        world.append(r)

    r = IdealRobot(np.array([0,0,0]).T, sensor=None, agent=circling, color="red")
    world.append(r)

    world.draw()


def noise_sensor_pattern():
    world = World(50, 0.1)

    m = Map()
    m.append_landmark(Landmark(-4,2))
    m.append_landmark(Landmark(2,-3))
    m.append_landmark(Landmark(-5,1))
    m.append_landmark(Landmark(-1,1))
    m.append_landmark(Landmark(-1,-1))
    m.append_landmark(Landmark(4,3))
    m.append_landmark(Landmark(2,2))
    m.append_landmark(Landmark(4,4))
    m.append_landmark(Landmark(3,3))
    world.append(m)

    circling = Agent(0.5, 10.0/180*math.pi)
    r = Robot(np.array([0,0,0]).T, 
              sensor=Camera(m, phantom_prob=0.01, occlusion_prob=0.01), 
              agent=circling,
              expected_kidnap_time=10,
              )
    world.append(r)

    world.draw()


if __name__ == "__main__":
    # random_init_pattern()
    # bias_control_pattern()
    # stuck_pattern()
    # kidnap_pattern()
    noise_sensor_pattern()