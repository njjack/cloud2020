import numpy as np
import math
from operator import attrgetter
import RL2
import copy
import torch
import random
import datetime
# how much future time in scheduling
TIMESLOT = 20

# resource capacity in a unit time
CAPACITY = 10

# number of resource type
R_TYPE = 2

# number of job slot in queue
Q_SIZE = 10

# column of backlog
BACKLOG_C = 4

# range of job set size [a, b)
JOBSET_SRANGE = (5, 51)

# job arrival rate (exponential)
#JOB_RATE = 2.0
JOB_RATE = 0.7

class Cluster:
    def __init__(self):
        self.schedule = [[0 for i in range(TIMESLOT)] for j in range(R_TYPE)]
        self.data = np.zeros((R_TYPE, TIMESLOT, CAPACITY), dtype=np.int)

    def insertJob(self, job):
        ok = False
        t = 0
        while t + job.time <= TIMESLOT:
            next_t = -1
            for shift in range(job.time):
                for r in range(R_TYPE):
                    if self.schedule[r][t+shift] + job.resource[r] > CAPACITY:
                        next_t = t + shift + 1
            
            if next_t == -1:
                ok = True
                break
            t = next_t

        if ok == False:
            return False, -1
        
        for shift in range(job.time):
            for r in range(R_TYPE):
                self.data[r][t+shift][self.schedule[r][t+shift]:self.schedule[r][t+shift]+job.resource[r]] = 1
                self.schedule[r][t+shift] += job.resource[r]
        return True, t

    def increaseTime(self):
        for time in range(TIMESLOT-1):
            for r in range(R_TYPE):
                self.data[r][time] = self.data[r][time+1]
                self.schedule[r][time] = self.schedule[r][time+1]
        for r in range(R_TYPE):
            self.data[r][TIMESLOT-1] = 0
            self.schedule[r][TIMESLOT-1] = 0
        return

    def isEmpty(self):
        return (not np.any(self.data))

    def show(self):
        print(self.data)
        return

class Job:
    def __init__(self, duration, demand):
        self.time = duration
        self.resource = demand
        self.data = np.zeros((R_TYPE, TIMESLOT, CAPACITY), dtype=np.int)
        self.enterTime = 0
        for i in range(R_TYPE):
            for j in range(self.time):
                for k in range(self.resource[i]):
                    self.data[i][j][k] = 1
    
    def isEmpty(self):
        if self.time == 0:
            return True
        return False

    def show(self):
        print(self.data)
        return

class JobSet:
    def __init__(self, select_function):
        self.jobnumber = np.random.randint(JOBSET_SRANGE[0], JOBSET_SRANGE[1])
        self.ctr = Cluster()
        self.jobqueue = []
        self.backlog = np.zeros((TIMESLOT, BACKLOG_C), dtype=np.int)
        self.backlogEnterTime = np.zeros((TIMESLOT, BACKLOG_C), dtype=np.int)
        self.backlog_cnt = 0
        self.in_queue_cnt = 0        
        self.allqueue = [0.0]
        self.TIME = 0
        self.PICK = select_function
        self.totalCyclingTime = 0
        for i in range(Q_SIZE):
            t, r = genJobParam()
            self.jobqueue.append(Job(0, r)) # empty job
        for i in range(self.jobnumber):
            self.allqueue.append(self.allqueue[-1]+np.random.exponential(scale = 1/JOB_RATE))            
            #self.allqueue.append(self.allqueue[-1]+np.random.exponential(scale=JOB_RATE))
        self.allqueue.pop(0)

    def pickNext(self, isRL):
        for i in range(Q_SIZE):
            if self.jobqueue[i].isEmpty() and self.backlog_cnt > 0:
                brow = int((self.backlog_cnt-1)/BACKLOG_C)
                bcol = int((self.backlog_cnt-1)%BACKLOG_C)
                self.backlog[brow][bcol] = 0
                self.backlog_cnt -= 1
                self.in_queue_cnt += 1
                t, r = genJobParam()
                self.jobqueue[i] = Job(t, r)
                self.jobqueue[i].enterTime = self.backlogEnterTime[brow][bcol]

        idx = -1
        
        if isRL:
            idx_list = self.PICK(self.getState())[0]
            if idx_list[0] == len(self.jobqueue): 
                print("no pick")
                return len(self.jobqueue), 0 # no job picked, also valid 
            
            for i in idx_list:
                if( i != len(self.jobqueue) ):
                    if( not self.jobqueue[i].isEmpty() ):
                        idx = i
                        break

            if( idx == -1):
                # print("no pick")
                return len(self.jobqueue), 0 # no job picked, also valid
        '''
        if isRL:
            idx = int(self.PICK(self.getState()))
            # print(idx)
            if idx == len(self.jobqueue): 
                return idx, 0 # no job picked, also valid 

        else:
            idx = self.PICK(self.jobqueue)
        
        if self.jobqueue[idx].isEmpty():
            return idx, 1
        
        '''
        insertRes, insertTime = self.ctr.insertJob(self.jobqueue[idx])
        if insertRes:
            jobEndTime = insertTime + self.TIME + self.jobqueue[idx].time
            self.totalCyclingTime += jobEndTime - self.jobqueue[idx].enterTime
            t, r = genJobParam()
            self.jobqueue[idx] = Job(0, r)
            self.in_queue_cnt -=1
        else:
            return idx, 1
        return idx, 0

    def increaseTime(self):
        self.ctr.increaseTime()
        self.TIME += 1
        while self.allqueue:
            if self.TIME > self.allqueue[0]:
                r = int((self.backlog_cnt)/BACKLOG_C)
                c = int((self.backlog_cnt)%BACKLOG_C)
                self.backlog[r][c] = 1
                self.backlogEnterTime[r][c] = self.TIME
                self.backlog_cnt += 1
                self.allqueue.pop(0)
            else:
                break
        return

    def getState(self):
        result = np.zeros((TIMESLOT, R_TYPE*CAPACITY*(Q_SIZE+1) + BACKLOG_C), dtype=np.int)
        for r in range(R_TYPE):
            result[:, (CAPACITY*r):(CAPACITY*(r+1))] = self.ctr.data[r]
        for j in range(Q_SIZE):
            for r in range(R_TYPE):
                result[:, (CAPACITY*R_TYPE + CAPACITY*j*R_TYPE + CAPACITY*r):(CAPACITY*R_TYPE + CAPACITY*j*R_TYPE + CAPACITY*(r+1))] = self.jobqueue[j].data[r]
        result[:, CAPACITY*R_TYPE*(Q_SIZE+1):] = self.backlog
        return result
    
    def isEmpty(self):
        if not self.ctr.isEmpty():
            return False
        for i in range(Q_SIZE):
            if not self.jobqueue[i].isEmpty():
                return False
        if np.any(self.backlog):
            return False
        if self.allqueue:
            return False
        return  True
    
    def get_reward(self):
        return self.in_queue_cnt + self.backlog_cnt

 
def genJobParam():
    # time
    if random.random() > 0.8:
        time = random.randint(10, 15)
    else:
        time = random.randint(1,3)

    # resource
    resource = []
    resource.append(random.randint(0.5*CAPACITY, CAPACITY))
    if random.random() > 0.5:
        resource.append(random.randint(0.1*CAPACITY, 0.2*CAPACITY))
    else:
        resource.insert(0, random.randint(0.1*CAPACITY, 0.2*CAPACITY))

    return time, resource

def SJF(JOB_QUEUE):
    idx = 0
    minTime = TIMESLOT + 1
    for i in range(len(JOB_QUEUE)):
        if JOB_QUEUE[i].time > 0 and JOB_QUEUE[i].time < minTime:
            idx = i
            minTime = JOB_QUEUE[i].time

    return idx
        
def RANDOM(JOB_QUEUE):
    return np.random.randint(0, Q_SIZE)

def train_RL(iteration = 15, gamma = 0.99, M = 20, jobset_cnt = 25, policy = None):#20,25
    agent = RL.RLagent()

    #agent.policy = torch.load('/Users/mcnlab/Downloads/pcs/RLmodel8_2_10')

    #agent.policy = torch.load('RLmodel')
    for a in range(8):
        jobset = []
        for i in range(jobset_cnt):
            jobset.append(JobSet(agent.select_action_train))
        
        for i in range(iteration):
            tmp2 = 0.0
            #print(i, ":", datetime.datetime.now())
            agent.optimizer.zero_grad()
            for J in jobset:
                R = []
                log_prob = []
                tmp = 0.0
                for m in range(M):
                    r = []
                    penalty = 0
                    j = copy.deepcopy(J)
                    j.PICK = agent.select_action_train

                    while len(r) < j.jobnumber:
                        j.increaseTime()
                        while 1:
                            action, invalid = j.pickNext(1)
                            reward = j.get_reward()
                            r.append(reward)
                            if j.in_queue_cnt != 0 and invalid ==1:
                                penalty += 1
                                reward += penalty
                            else:
                                penalty = 0

                            
                            if (action == 10) or invalid:
                                break

                    r = r[:j.jobnumber]
                    tmp += sum(r)
                    for t in range(len(r)-2,-1,-1):
                        r[t] = r[t] + r[t+1]*gamma

                    R.append(r)
                    log_prob.append(agent.log[:j.jobnumber])
                    agent.refresh_traj()
                
                R = torch.FloatTensor(R)
                b = torch.mean(R,0)
                tmp /= M

                for m in range(M):
                    loss = torch.sum(torch.mul(log_prob[m], R[m]-b))
                    loss.backward()
            tmp2 += tmp #torch.mean(R).item()
            print(i,":", tmp2, datetime.datetime.now())
            #print(i,":", tmp, datetime.datetime.now())
            agent.optimizer.step()
        
    return agent


if __name__ == '__main__':
    np.random.seed()
    
    '''
    agent = train_RL(iteration = 100)
    torch.save(agent.policy, 'RLmodel')
    ''' 
    agent = RL2.RLagent(False)
    agent.policy = torch.load("RLmodel9_15*8")
    if (torch.cuda.is_available()):
        agent.policy = agent.policy.to('cuda')
    else:
        print("cuda not available, use cpu")

    agent.policy.eval()

    jobset_cnt = 200
    res = 0
    print("start")
    for i in range( jobset_cnt ):
        js = JobSet(agent.select_action_eval)
        while not js.isEmpty():
            js.increaseTime()
            action, invalid = js.pickNext(1)
            # if( invalid ):
            #     print(f'js.jobnumber = {js.jobnumber}, action = {action}, js.in_queue_cnt={js.in_queue_cnt}, js.backlog_cnt = {js.backlog_cnt}')
            #     js.jobqueue[action].show()
                
        print( js.totalCyclingTime / js.jobnumber )
        
        res+=js.totalCyclingTime / js.jobnumber
    print("result:", res/ jobset_cnt)


    # Sample Simulation

    # Create a jobset, which contains cluster, job queue, backlog and a queue which
    # records arrival time of all jobs.
    # The only parameter is the decision function when selecting job from queue

    # JobSet.increastTime() increase time by 1 and will modify the cluster, 
    # storing new arriving jobs into backlog(if any).


    # JobSet.pickNext() is the action which picks out a job from job queue. Before making decision,
    # it will load available jobs from backlog into job queue. You can call this method multiple times
    # in a unit of time.
    
    # The only parameter is the type of decision function. If you are using NN/RL, please set it to 1.
    # If you are using SJF or RANDOM function, please set to 0.

    # Return Value of pickNext() is the index of selected job in jobqueue, if an empty job is selected
    # or the selected job cannot be added into cluster, -1 is returned.
    '''
    modify pickNext() for returning index and invalid. 
    invalid: 1 means invalid, 0 means valid
    When training rl model, the action has to be record even it's invalid.
    '''

    # JobSet.getState() returns the 2D Numpy array of state with size
    # being (TIMESLOT, R_TYPE*CAPACITY*(Q_SIZE+1) + BACKLOG_C) for training model.
    # For example, if TIMESLOT=20, R_TYPE=2, CAPACITY=10, Q_SIZE=10, BACK_C=4,
    # the size is 20*224, which coincide the setting in paper.

    '''
    get_reward(): return the total count of jobs in queue and backlog
    '''

    '''
    in_queue_cnt: jobs count in jobqueue
    '''

    '''
    modify genJobParam(), jobrate, and line112 for fitting the workload design in paper
    '''