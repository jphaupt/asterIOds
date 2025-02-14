#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 13:30:11 2018

@author: jph

asteroids game
enhance later to by played by an NN with RL 
plan: use polar coordinates and maybe restrict sight so that it "simulates"
someone actually in the ship, as opposed to just raw pixels 
add screening? some number of closest asteroids? idk
would probably be easier to use NEAT but it could be neat (geddit) to try RL
and see how the NN learns *mid-game*

this file is just for the game itself though - baby steps!! 
maybe even first restrict the line of vision for the human player to see if 
it's doable

if I can't figure out RL, I'll use NEAT (but I don't wanna!)

for now, sprites are just circles and a triangle

maybe add alien (hostile) spaceships later? 

maybe include pickups (e.g. lives) 
this has no immediate benefit, but long-term benefit, so maybe the network can
learn to use it

TODO : documentation you dunce! 

TODO : collisions! 
"""

# %% import core packages
import pygame
import numpy as np
import random
import math
import pygame.locals as pgl
import neat
#import sys, os



# %% global values
# constants
# game height and width in pixels
CANVAS_HEIGHT = 1000
CANVAS_WIDTH = 1000
MIN_ROCK_RADIUS = 15
SHIP_WIDTH = 6 * 3 # TODO : still not sure about ship size
SHIP_HEIGHT = 10 * 3
BLACK = (0,0,0)
WHITE = (255,255,255) 
GREEN = (0, 255, 0) 
RED = (255, 0, 0)
BLUE = (0, 0, 255) 
NUM_PER_SPLIT = 3 # number of rocks a rock splits into when hit
MAX_MISSILES = 3 # limit number of missiles possible to shoot 
#visualize = False # figure out way to handle this outside file 
SCORE_FOR_ROCK = 30
SCORE_FOR_CLEARING = 10

#seed = 7  # for reproducibility (i.e. training) 
seed = None # for validation testing 
#print(seed) 

# %% classes for objects
class Sprite : 
    def __init__(self, pos, vel):
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        
    def update(self) :
        self.pos += self.vel 
        self.pos[0] %= CANVAS_WIDTH
        self.pos[1] %= CANVAS_HEIGHT
    # TODO stub

# ship class (i.e. the player) 
# TODO : stub
# TODO : make sure Ship has acceleration (unlike others) 
# TODO : should probably make Ship inherit Sprite?
class Ship(Sprite) :
    def __init__(self, pos, vel, ang):
        Sprite.__init__(self, pos, vel) 
        self.ang = ang # angle 
        self.ang_vel = 0
        self.thrust = False
        self.accel = 0
        self.front = angle_to_vector(self.ang) 
        
    def update(self) :
#        self.ang += self.ang_vel
#        self.front = angle_to_vector(self.ang) 
#        
#        # TODO : decide on how to handle thrust
#        if not self.thrust:
#            self.vel *= 0.9
#        if np.linalg.norm(self.vel) <= 9:
#            self.vel += self.accel 
        
        acc = 0.1
        fric = acc / 20
        
        self.ang += self.ang_vel
        self.ang %= 2*np.pi 

        self.front = angle_to_vector(self.ang)

        if self.thrust:
            self.vel += self.front * acc

        self.vel *= (1 - fric)
        
        Sprite.update(self) 
        
    def set_ang_vel(self, ang_vel) :
        self.ang_vel = ang_vel 
        
    def set_thrust(self, thrust) :
        self.thrust = thrust
        
    def shoot(self) :
        if len(missiles) < MAX_MISSILES :
            missile_pos = self.front + self.pos
            missile_vel = self.vel + 5 * self.front 
            missiles.append(Missile(missile_pos, missile_vel))  
    
class Rock(Sprite) :
    '''
    class for the asteroids floating around, to be shot for points
    '''
    def __init__(self, pos, vel, size=3) :
        Sprite.__init__(self, pos, vel) 
        self.size = size
    # TODO stub

class Missile(Sprite) : 
    '''
    missile class for when the player hits spacebar
    '''
    def __init__(self, pos, vel, age=0) : 
        Sprite.__init__(self, pos, vel) # TODO : do I include self as param here?  
        self.age = age # missiles disappear after some time, even if it doesn't hit anything
        self.lifespan = 60 # TODO : decide on number 
        
    def update(self) :
        self.age += 1
        Sprite.update(self)
        return self.age <= self.lifespan
#        print("STUB")
    # TODO stub 
# %% functions
def collide_missile_rock(rocks, missiles, score) :
    '''
    check collision of any rock with any missile and update (decrease size) 
    rock and delete missile if true
    returns number of hits 
    '''
#    global score
    hits = 0 
    for rock in rocks :
        for missile in missiles :
            if rock in rocks : # since several might hit at same time 
                if np.linalg.norm(rock.pos - missile.pos) <= rock.size * MIN_ROCK_RADIUS:
                    if rock.size > 1:
                        # TODO : SPLIT INTO 3 (?) OTHER ROCKS
                        new_rocks = list([]) 
                        for i in range(NUM_PER_SPLIT-1) :
                            new_vel = [random.uniform(-2, 2), random.uniform(-2, 2)]
                            new_rock = Rock(rock.pos, new_vel, size=rock.size-1)
                            new_rocks.append(new_rock) 
                        # conservation of momentum, because I'm a huge fucking nerd
                        new_vel = rock.vel.copy()
    #                    print(new_vel) 
                        for new_rock in new_rocks :
    #                        print(new_rock.vel) 
                            new_vel -= new_rock.vel
                        new_rocks.append(Rock(rock.pos, new_vel, size=rock.size-1)) 
                        rocks.remove(rock) 
                        rocks.extend(new_rocks) 
                        missiles.remove(missile)
                        hits += 1
#                        print("hit!") 
                    else : 
                        rocks.remove(rock) 
                        missiles.remove(missile)
                        hits += 1 
                        if len(rocks) == 0:
                            hits += SCORE_FOR_CLEARING # bonus points for clearing level... Why not?
                            spawn_random_rocks(player, score) # neeeext
                            pygame.time.set_timer(SPAWN_ROCKS, t) 
    return hits
    pass 

def collide_ship_rock(ship, rocks) :
    '''
    check collision of any rock with the ship and update lives if true
    todo : also something about respawn and temporary invulnerability? 
    '''
    # TODO : actually check if the SHIP collides 
    # at the moment, only checks if the pixels in corners collide,
    # not the triangle :( 
    # TODO : do stuff in the case of still having lives left
    # get ship points, and check if they are in the circle 
    # this is not the same as checking triangle collision (just a temp fix)
    # also checking cockpit, just for good measure
#    print('check collision')
    rot = rotate_mat(ship.ang)
    pts2check = [ship.pos-rot.dot(np.array([SHIP_WIDTH//2, SHIP_HEIGHT//2])), 
                                 ship.pos-rot.dot(np.array([-SHIP_WIDTH//2, SHIP_HEIGHT//2])), 
                                 ship.pos+rot.dot(np.array([0, SHIP_HEIGHT//2])),
                                 ship.pos]
    for rock in rocks :
        for pt in pts2check :
            if np.linalg.norm(rock.pos - pt) <= rock.size * MIN_ROCK_RADIUS:
                return True
        
    return False

# draw on canvas (also updates and does collision check!)
def draw(canvas, score, visualize=True) :
    global lives
#    print('in draw')
#    score = 0
    if visualize :
#        print(visualize)
        canvas.fill(BLACK)
        # print lives and score
        fontsize = 15
        text1 = "lives: %i" % lives
        text2 = "score: %i" % score
        text_font = pygame.font.Font('freesansbold.ttf',fontsize)
        text_surface = text_font.render(text1, True, WHITE)
        text_rect = text_surface.get_rect()
        text_rect.center = ((CANVAS_WIDTH-50, fontsize+5))
        canvas.blit(text_surface, text_rect)
        text_surface = text_font.render(text2, True, WHITE)
        text_rect = text_surface.get_rect()
        text_rect.center = ((50, fontsize+5))
        canvas.blit(text_surface, text_rect)
#    update_rocks(canvas)
    
    # draw player
    if is_running : 
        player.update() 
        if visualize :
            rot = rotate_mat(player.ang)
            pygame.draw.polygon(canvas, GREEN, 
                                [player.pos-rot.dot(np.array([SHIP_WIDTH//2, SHIP_HEIGHT//2])), 
                                 player.pos-rot.dot(np.array([-SHIP_WIDTH//2, SHIP_HEIGHT//2])), 
                                 player.pos+rot.dot(np.array([0, SHIP_HEIGHT//2]))], 
                                 1)
            # person in cockpit, represented by a single pixel (lol) 
            pygame.draw.line(canvas, WHITE, player.pos, player.pos)
        
        # check rock-missile collision
        hits = collide_missile_rock(rocks, missiles, score) 
        # update score accordinly
        if hits > 0:
            score += hits 
            
        if collide_ship_rock(player, rocks) :
            lives -= 1
#            print("collided")
        
    #    print(player.pos)
    #    pygame.draw.line(canvas, (255, 0, 0), player.pos + 5*player.front, player.pos + 5*player.front)
        # TODO : collision detection
        # draw asteroids
        for rock in rocks :
            rock.update() 
            if visualize : 
                pygame.draw.circle(canvas, WHITE, [int(rock.pos[0]), int(rock.pos[1])], MIN_ROCK_RADIUS * rock.size, 1)
                
        for missile in missiles :
            if missile.update() :
                # TODO : do this in a prettier and maybe more efficient way :( 
                if visualize :
                    pygame.draw.circle(canvas, RED, [int(missile.pos[0]), int(missile.pos[1])], 1, 0)
        #        pygame.draw.line(canvas, RED, missile.pos, missile.pos)
            else : 
                missiles.remove(missile) 

    if lives < 1:
        game_over(visualize=visualize)
        
    return score
    
def rotate_mat(angle) :
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])
    return rot 

def keydown(event) :
    ang_vel = 3*np.pi/180
    
    if event.key == pgl.K_LEFT:
        player.set_ang_vel(-ang_vel)
    if event.key == pgl.K_RIGHT:
        player.set_ang_vel(ang_vel)
    if event.key == pgl.K_UP:
        player.set_thrust(True)
    if event.key == pgl.K_SPACE:
        player.shoot()
    
def keyup(event) :
    if event.key in (pgl.K_LEFT,pgl.K_RIGHT):
        player.set_ang_vel(0)
    if event.key == pgl.K_UP:
        player.set_thrust(False)

def angle_to_vector(ang):
    return np.array([-math.sin(ang), math.cos(ang)])

def game_over(visualize=True) : 
    '''
    run when lives reaches zero. Ends game and displays a message
    might need to be modified for the neural network
    '''
    global is_running
    if visualize :
        fontsize = 100
        text1 = "THANKS"
        text2 = "OBAMA"
        text_font = pygame.font.Font('freesansbold.ttf',fontsize)
        text_surface = text_font.render(text1, True, WHITE)
        text_rect = text_surface.get_rect()
        text_rect.center = ((CANVAS_WIDTH//2, CANVAS_HEIGHT//2-fontsize))
        canvas.blit(text_surface, text_rect)
        text_surface = text_font.render(text2, True, WHITE)
        text_rect = text_surface.get_rect()
        text_rect.center = ((CANVAS_WIDTH//2, CANVAS_HEIGHT//2+fontsize))
        canvas.blit(text_surface, text_rect)
    is_running = False # not sure if I like this
    # TODO : determine how long to keep on, what to do after

def spawn_random_rocks(ship, score) :
#    global score
    # TODO : make dependent on score? spawn several at once, random locations
    # but not on top of the ship
#    print("Spawning rocks")
    # not sure what number to divide score by 
    # TODO : seems NNs have no idea what to do when there is less than 5 rocks
    # how many rocks do I spawn? make func tion of score? make random?
    num_to_spawn = int(score/SCORE_FOR_ROCK) + 2#random.randint(1,4) 
    for i in range(num_to_spawn) : 
        ang = random.uniform(0, 2 * np.pi)
        r = random.uniform(SHIP_HEIGHT * 5, CANVAS_WIDTH - SHIP_HEIGHT * 5)
        position = ship.pos + r * angle_to_vector(ang)
        position[0] %= CANVAS_WIDTH
        position[1] %= CANVAS_HEIGHT
        rock = Rock(position, [random.uniform(-2,2), random.uniform(-2,2)]) 
        rocks.append(rock) 

def closest_vect(x,y) : 
    # TODO : not 100% sure if this is working as intended
    # seems to be fine though! I think I fixed it 
    dx = x.pos[0] - y.pos[0]
    if dx > CANVAS_WIDTH/2:
        dx = dx - CANVAS_WIDTH 
    elif dx < -CANVAS_WIDTH/2:
        dx = CANVAS_WIDTH - dx
    dy = x.pos[1] - y.pos[1]
    if dy > CANVAS_HEIGHT/2:
        dy = dy - CANVAS_HEIGHT
    elif dy < -CANVAS_HEIGHT/2:
        dy = CANVAS_HEIGHT - dy
    return [dx,dy]

def dist_sq_bw(x, y) :
    '''
    calculate square distance between two sprites
    '''
    dx, dy = closest_vect(x,y) 
#    dx = abs(dx)
#    dy = abs(dy)
#    if dx > CANVAS_WIDTH/2:
#        dx = CANVAS_WIDTH - dx
#    if dy > CANVAS_HEIGHT/2:
#        dy = CANVAS_HEIGHT - dy
    return dx**2 + dy**2

#def ang_bw(x, y) :
#    '''
#    calculates the angle between two vectors
#    '''
#    lenX = np.linalg.norm(x)
#    lenY = np.linalg.norm(y)
#    ret_ang = np.arccos(np.dot(x,y)/(lenX*lenY))
#    print(ret_ang)
#    return ret_ang

def find_neural_input(nn, ship, rocks, visualize) :
    '''
    determine neural network input : 
        size and location (polar coords) of 5 closest rocks
    TODO : should I also give informaton about the # of missiles fired or 
    similar? 
    
    dx = abs(x1 - x2);
    if (dx > width/2)
        dx = width - dx;
    // again with x -> y and width -> height
    
    NOTE : inputs are normalised as follows 
        square distance divided by CANVAS_WIDTH*CANVAS_HEIGHT/2
        angle / pi (so between -1 and 1)
        size / 3 TODO : improve this one...
        
    '''
    nn_in = np.zeros(nn.nb_input,) # TODO : ones or zeros? Something else? 
#    print(nn_in.shape) 
    # sticking with zeros for now because it's the only one that seems to give
    # actually interpretable movement (although not the best fitness) 
    dist_ind = np.arange(0,neat.NUM_ROCK_IN)*neat.NUM_IN_PER_ROCK
    ang_ind = np.arange(0,neat.NUM_ROCK_IN)*neat.NUM_IN_PER_ROCK + 1
    # TODO : decide whether or not to keep the size as input 
    if neat.NUM_IN_PER_ROCK >= 3 : 
        size_ind = np.arange(0,neat.NUM_ROCK_IN)*neat.NUM_IN_PER_ROCK + 2
    # TODO : initialize big distances?? 
    # initalize distances to a large number (getting NaN with np.inf)
#    nn_in[dist_ind] = 100*np.max((CANVAS_WIDTH, CANVAS_HEIGHT))
    rocks.sort(key = lambda x : dist_sq_bw(x, ship)) 
    if len(rocks) < neat.NUM_ROCK_IN : 
        closest = rocks # TODO : complete implementing input 
    else :
        closest = rocks[:neat.NUM_ROCK_IN]
    for i in range(len(closest)) :
#        nn_in[dist_ind[i]] = dist_sq_bw(closest[i], ship) / (CANVAS_WIDTH*CANVAS_HEIGHT/4) # TODO : uncomment after debug
        
        r_sq = dist_sq_bw(closest[i], ship)
#        ang = (nn_in[1] + player.ang)
#        pygame.draw.line(canvas, BLUE, player.pos, player.pos + r*angle_to_vector(ang))
        nn_in[dist_ind[i]] = r_sq / (CANVAS_WIDTH * CANVAS_HEIGHT / 4) # normalize
#        print(nn_in[dist_ind[i]]) 
        ship2rock = closest_vect(closest[i], ship)
        # rotate by ship.ang 
        # TODO : think I'm messing this up
#        ship2rock[0] %= CANVAS_WIDTH / 2
#        ship2rock[1] %= CANVAS_HEIGHT / 2
        rot = rotate_mat(-ship.ang+np.pi)
        ship2rock = rot.dot(ship2rock) 
        # TODO : not sure if this is a good way to get polar coords
#        ship2rock = closest[i].pos - ship.pos
#        ship2rock[0] %= CANVAS_WIDTH / 2
#        ship2rock[1] %= CANVAS_HEIGHT / 2
        ret_ang = math.atan2(ship2rock[0], -ship2rock[1]) # - (ship.ang-np.pi) 
        if visualize and is_running:    
            r = np.sqrt(r_sq) 
            pygame.draw.line(canvas, BLUE, player.pos, player.pos + r*angle_to_vector(ret_ang+ship.ang)) 
        
#        if ret_ang < -np.pi:
#            ret_ang += 2*np.pi
#        elif ret_ang > np.pi:
#            ret_ang -= 2*np.pi 
        
        ret_ang /= np.pi  
        nn_in[ang_ind[i]] = ret_ang
#        print(nn_in[ang_ind[i]])
        if neat.NUM_IN_PER_ROCK >= 3 :
#            print(size_ind) 
            nn_in[size_ind[i]] = closest[i].size / 3
    if visualize and is_running: 
        pygame.draw.line(canvas, WHITE, player.pos, player.pos) 
    nn_in[-1] = len(missiles) / MAX_MISSILES
    return nn_in

# %% game loop
def game_loop(isAI=False, nn=None, visualize=True) :
    global canvas, player, time, lives, missiles, rocks, is_running, SPAWN_ROCKS, t
    if isAI : 
        random.seed(seed) 
    pygame.init()
    if visualize :
        canvas = pygame.display.set_mode((CANVAS_WIDTH,CANVAS_HEIGHT))
        pygame.display.set_caption('Asteroids')
    if not visualize :
        canvas = None # pass in dummy object since we're not drawing anything
    clock = pygame.time.Clock()
    
    # variable
    time = 0
    score = 0
    lives = 1 # maybe add more lives? idk probs 1 at least for RL 
    is_running = True
    
    
    # TODO : I have absolutely not idea what kind of data types to use here...
    rocks = list([])
    missiles = list([]) 
    
    SPAWN_ROCKS, t = pygame.USEREVENT+1, 30000 # how often (ms) to spawn random rocks
    pygame.time.set_timer(SPAWN_ROCKS, t) 
    player = Ship([CANVAS_WIDTH / 2,CANVAS_HEIGHT / 2], [0.,0.], np.pi)
    # TODO : make the first rock intentionally attack the player? 
    # so that the network doesn't just stand in place
    first_rock = Rock([0., CANVAS_HEIGHT/2], [1., 0.]) 
    rocks.append(first_rock) 
#    second_rock = Rock([CANVAS_WIDTH, 0.], [-1., -1.]) 
#    rocks.append(second_rock) 
#    spawn_random_rocks(player, score) 
    spawn_random_rocks(player, score) 
    # TODO : wtf is going on with the angle NN input?! 
#    spawn_random_rocks(player, score)
    # TODO : delete following line (just for debugging) 
#    nn = neat.Individual(neat.relu) 
    # TODO : do this in neat.py... somehow
#    tmp = False
    #player.set_ang_vel(np.pi/180)
    while is_running or visualize :
#        print('in loop')
#        tmp_score = score
        score = draw(canvas, score, visualize=visualize)
#        if tmp_score < score :  
#            print(score) 
#        if tmp :
#            print('should not be here...') 
        # AI player 
        # TODO make AI play this, obviously not just random input each time... 
        # TODO : different methods... 
    #    tmp_in = np.random.rand(15,)
    # TODO : restrict to only one action per frame? 
#        nn_in = find_neural_input(nn, player, rocks, visualize)
#        print(nn_in[:2])
        if isAI :
            nn_in = find_neural_input(nn, player, rocks, visualize)
                 
            ##    print(tmp_in) 
            nn_out = nn.predict(nn_in)  
            # allow several buttons to be pressed at once
            ang_vel = 3*np.pi/180
            # TODO : decide if this is really the best idea... 
#            if nn_out[0] >= 0 and nn_out[1] >= 0:
#                player.set_ang_vel(0) # if you press both, it does nothing
#            print(nn_out) 
            if nn_out[0] >= 0.5 and nn_out[0] > nn_out[1] : # left
#                    print('left')
                player.set_ang_vel(-ang_vel)
            elif nn_out[1] >= 0.5 and nn_out[1] > nn_out[0] : # right
#                    print('right')
                player.set_ang_vel(ang_vel)
            else :
                player.set_ang_vel(0) 
            if nn_out[2] >= 0.5 : # forward
#                    print('thrust') 
                player.set_thrust(True)
            else : 
                player.set_thrust(False)
            if nn_out[3] >= 0.5 : # should I have an else here?? 
#                    print('shoot')
                player.shoot() 
            #    print(tmp_out)
            #    print(player.pos) 
        
        if visualize :
            pygame.display.update()
            # limit FPS
            clock.tick(60) # not sure if worked 
    #    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # TODO : figure out why closing this gives errors 
#                print(score) 
#                if visualize : 
#                    pygame.display.quit()
#                    print('here') 
                pygame.quit()
#                os._exit(0) 
#                tmp = True
                # TODO : figure out why it keeps giving an error 
#                quit(keep_kernel=True)
            elif not isAI :
                if event.type == pygame.KEYDOWN:
                    keydown(event) 
                elif event.type == pygame.KEYUP: 
                    keyup(event) 
            elif event.type == SPAWN_ROCKS :
                spawn_random_rocks(player, score)
                
        
    
        if not visualize :
            if not is_running :
#                print(score) 
                return score
            
# %% if you just run the file
if __name__ == "__main__":
#    try :  
    performance = game_loop()
    print(performance)
#    except SystemExit : 
#        pass