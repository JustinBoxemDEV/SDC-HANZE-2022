# Implementation of the 2dracing game with the gym library

from re import I
from turtle import window_height, window_width
import pygame
import time
import math
from utils import scale_image, blit_rotate_center
pygame.font.init()
import numpy as np
import gym

# Some images to make it look pretty
GRASS = scale_image(pygame.image.load("src/MachineLearning/2dracing/imgs/grass.jpg"), 2.5)
TRACK = scale_image(pygame.image.load("src/MachineLearning/2dracing/imgs/track.png"), 0.9)
TRACK_BORDER = scale_image(pygame.image.load("src/MachineLearning/2dracing/imgs/track-border.png"), 0.9)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)
FINISH = pygame.image.load("src/MachineLearning/2dracing/imgs/finish.png")
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (130, 250)
RED_CAR = scale_image(pygame.image.load("src/MachineLearning/2dracing/imgs/red-car.png"), 0.55)
GREEN_CAR = scale_image(pygame.image.load("src/MachineLearning/2dracing/imgs/grey-car.png"), 0.55)

FPS = 60
window_width, window_height = TRACK.get_width(), TRACK.get_height()
rotation_max, acceleration_max = 0.08, 0.5

WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
MAIN_FONT = pygame.font.SysFont("Comic Sans", 44)

class AbstractCar:
    IMG = RED_CAR
    START_POS = (150, 200)

    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x, self.y = self.START_POS

    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.y -= vertical
        self.x -= horizontal

    # Not used yet
    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        return poi

    def reset(self):
        self.x, self.y = self.START_POS
        self.angle = 0
        self.vel = 0

class PlayerCar(AbstractCar):
    # Change the colour/start position of the car here
    IMG = RED_CAR
    START_POS = (150, 200)

    # Not used yet
    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()

    # Not used yet
    def bounce(self):
        self.vel = -self.vel *0.5
        self.move()

class ComputerCar(AbstractCar):
    # Change the colour/start position of the car here
    IMG = GREEN_CAR
    START_POS = (150, 200)

class Custom2dRacingHuman(gym.Env):
    def __init__(self, env_config={}):
        self.car = PlayerCar(4, 4)
        self.window = None
        self.clock = None

    def init_render(self):
        print("Init render")
        pygame.display.set_caption("2D Racing for Humans")
        self.window = pygame.display.set_mode((window_width, window_height))
        self.clock = pygame.time.Clock()

    def reset(self):
        print("Resetting environment")
        self.car.reset()
        observation = None # temporary
        return observation

    def step(self, action=np.zeros((2),dtype=float)):
        # print(f"Executing step with action {action}")
        
        # apply rotation
        self.car.angle = self.car.angle + self.car.rotation_vel * action[1]

        # apply acceleration (backwards acceleration at half thrust)
        if self.car.vel < 0:
            self.car.vel = max(action[0], -self.car.max_vel/2)
        else:
            self.car.vel = min(action[0], self.car.max_vel)

        # move car
        self.car.move()

        # temporary empty values
        observation, reward, done, info = 0., 0., False, {}

        return observation, reward, done, info

    def render(self):
        images = [(GRASS, (0, 0)), (TRACK, (0, 0)), (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]
        
        for img, pos in images:
            self.window.blit(img, pos)

        # Some values for debugging
        vel_text = MAIN_FONT.render(f"Vel: {round(self.car.vel, 1)}px/s", 1, (255, 255, 255))
        self.window.blit(vel_text, (5, HEIGHT - vel_text.get_height() - 10))
        rotation_vel_text = MAIN_FONT.render(f"Rota: {self.car.angle}", 1, (255, 255, 255))
        self.window.blit(rotation_vel_text, (5, HEIGHT - rotation_vel_text.get_height() - 60))

        self.car.draw(self.window)
        pygame.display.update()

class Custom2dRacingAI(gym.Env):
    def __init__(self, env_config={}):
        self.car = ComputerCar(4, 4)
        self.window = None
        self.clock = None

    def init_render(self):
        print("init render")
        pygame.display.set_caption("2D Racing for AI")
        self.window = pygame.display.set_mode((window_width, window_height))
        self.clock = pygame.time.Clock()

    def reset(self):
        print("resetting environment")
        self.car.reset()
        observation = None # temporary
        return observation

    def step(self, action=np.zeros((2),dtype=float)):
        # print(f"Executing step with action {action}")
        
        # apply rotation
        self.car.angle = self.car.angle + self.car.rotation_vel * action[1]

        # apply acceleration (backwards acceleration at half thrust)
        if self.car.vel < 0:
            self.car.vel = max(action[0], -self.car.max_vel/2)
        else:
            self.car.vel = min(action[0], self.car.max_vel)

        # move car
        self.car.move()

        # temporary empty values
        observation, reward, done, info = 0., 0., False, {}

        return observation, reward, done, info

    def render(self):
        images = [(GRASS, (0, 0)), (TRACK, (0, 0)), (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]
        
        for img, pos in images:
            self.window.blit(img, pos)

        # Some values for debugging
        vel_text = MAIN_FONT.render(f"Vel: {round(self.car.vel, 1)}px/s", 1, (255, 255, 255))
        self.window.blit(vel_text, (5, HEIGHT - vel_text.get_height() - 10))
        rotation_vel_text = MAIN_FONT.render(f"Rota: {self.car.angle}", 1, (255, 255, 255))
        self.window.blit(rotation_vel_text, (5, HEIGHT - rotation_vel_text.get_height() - 60))

        self.car.draw(self.window)
        pygame.display.update()

# For playing as human
def pressed_to_action(keytouple):
    action_turn = 0.
    action_acc = 0.
    if keytouple[pygame.K_UP] == 1:  # forward
        action_acc += 1
    if keytouple[pygame.K_DOWN] == 1:  # back
        action_acc -= 1
    if keytouple[pygame.K_LEFT] == 1:  # left  is -1
        action_turn += 1
    if keytouple[pygame.K_RIGHT] == 1:  # right is +1
        action_turn -= 1

    return np.array([action_acc, action_turn])

# Play as human
def runHuman():
    environment = Custom2dRacingHuman()
    environment.init_render()
    run = True
    while run:
        # set game speed to 30 fps
        environment.clock.tick(30)
        
        # end while-loop when window is closed
        get_event = pygame.event.get()
        for event in get_event:
            if event.type == pygame.QUIT:
                run = False

        # get pressed keys, generate action
        get_pressed = pygame.key.get_pressed()
        action = pressed_to_action(get_pressed)

        # calculate one step
        environment.step(action)

        # render current state
        environment.render()
    pygame.quit()

# Test loop for NN
def runTestAI():
    clock = pygame.time.Clock()
    env = Custom2dRacingAI()
    env.init_render()
    run = True
    # Load model here
    while run:
        clock.tick(30)
        
        # end while-loop when window is closed
        get_event = pygame.event.get()
        for event in get_event:
            if event.type == pygame.QUIT:
                run = False

        action = [0,0] # get agent prediction; model.predict()
        env.step(action)
        env.render()
        pygame.display.update()
    pygame.quit()


# Train loop for NN
def runTrainAI():
    total_reward = 0
    env = Custom2dRacingAI()
    env.init_render()
    epochs = 20
    batch_size = 100
    for i_episode in range(epochs):
        observation = env.reset()
        for t in range(batch_size):
            env.render()
            action = env.action_space.sample() # Random action for now, should be model.predict()
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
    env.close()

if __name__ == "__main__":
    runHuman()
    # runTrainAI() # not working yet
    # runTestAI() # always uses action 0,0
