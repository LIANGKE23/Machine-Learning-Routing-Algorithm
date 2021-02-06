import numpy as np
import random
from queue import Queue
import matplotlib.pyplot as plt

def Network_Topology_Init():
    network_topology = [[1, 6], [0, 2, 7], [1, 8], [4, 9], [3, 5, 10], [4, 11], [0, 7, 12], [1, 6, 8, 13], [2, 7, 14],
                        [3, 10, 15], [4, 9, 11, 16], [5, 10, 17], [6, 13, 18], [7, 12, 14, 19], [8, 13, 20],
                        [9, 16, 21],[10, 15, 17, 22], [11, 16, 23], [12, 19, 24], [13, 18, 20, 25], [14, 19, 21, 26],
                        [15, 20, 22, 27],[16, 21, 23, 28], [17, 22, 29], [18, 30], [19, 26], [20, 25], [21, 28], [22, 27],
                        [23, 35],[24, 31], [30, 32], [31, 33], [32, 34], [33, 35], [29, 34]]
    topo_size = len(network_topology)
    return network_topology,topo_size

def Perriodically_Introduced_Packet(currenttime,load_list,activenode_list,node_list,assert_num,topo_size):
    packets_list = []
    packets_count = load_list[currenttime]
    # packets_count = 3
    for x in range(packets_count):
        nodes = random.sample(range(0, topo_size), 2)
        packets_list.append([nodes[0], nodes[1], nodes[0], 0, 0, 0])
    num = len(packets_list)
    new_packet_list = list()
    if num > 0:
        for id in range(num):
            node_list[packets_list[id][0]].put(assert_num)
            activenode_list.append(assert_num)
            new_packet_list.append(packets_list[id])
    return activenode_list,node_list,new_packet_list

def Table_Init(Q_or_CQ,topo_size):
    Q_Table = []
    C_Table = []
    for i in range(topo_size):
        Q_Table.append(np.ones((topo_size, topo_size)))
        if Q_or_CQ != 'Q':
            C_Table.append(np.ones((topo_size, topo_size)))
    if Q_or_CQ == 'Q':
        return Q_Table
    else:
        return Q_Table,C_Table

def Experiments_Q_CQ_Routing(load,max_steps,Q_or_CQ,learn_rate,lamda):
    node_list = list()
    activenode_list = list()
    avg_list = list()
    packet_list = list()
    avg_nodes = 0
    avg_time = 0
    avg_count = 0
    time_count = 0
    rule3a_or_not = 0
    load_list = np.random.poisson(lam=load-1, size=max_steps)
    network_topology,topo_size = Network_Topology_Init()
    if Q_or_CQ == 'Q':
        Q_Table = Table_Init(Q_or_CQ,topo_size)
    else:
        Q_Table,C_Table = Table_Init(Q_or_CQ,topo_size)
    for i in range(36):
        node_list.append(Queue(maxsize=0))
    while (time_count < max_steps):
        assert_num = len(packet_list)
        activenode_list,node_list,new_packet_list = Perriodically_Introduced_Packet(time_count,load_list,activenode_list,node_list,assert_num,topo_size)
        for packet in new_packet_list:
            packet_list.append(packet)
        for node_index in range(len(node_list)):
            if node_list[node_index].empty() !=True:
                packet_index = node_list[node_index].get()
                if node_list[node_index].empty() !=True:
                    for active_node_id in range(len(activenode_list)):
                        if (activenode_list[active_node_id] != packet_index)&(packet_list[activenode_list[active_node_id]][2] == node_index):
                            packet_list[activenode_list[active_node_id]][3] = packet_list[activenode_list[active_node_id]][3] + 1
                            packet_list[activenode_list[active_node_id]][4] = packet_list[activenode_list[active_node_id]][4] + 1
                packet = packet_list[packet_index]
                pnode = packet[2]
                dst = packet[1]
                if Q_or_CQ == 'Q':
                    qtable = Q_Table[pnode]
                else:
                    qtable = Q_Table[pnode]
                    ctable = C_Table[pnode]
                greedy_num = random.random()
                if greedy_num <= 1:
                    for i in range(len(network_topology[pnode])):
                        if i == 0:
                            minq = 14000
                        if qtable[dst, network_topology[pnode][i]] <= minq:
                            minq = qtable[dst, network_topology[pnode][i]]
                            next_node = network_topology[pnode][i]
                else:
                    random_index = np.random.randint(0, len(network_topology[pnode]))
                    next_node = network_topology[pnode][random_index]
                next_ninq = 0
                if next_node != dst:
                    next_table = Q_Table[next_node]
                    for i in range(len(network_topology[next_node])):
                        if i == 0:
                            next_ninq = 14000
                        if next_table[dst, network_topology[next_node][i]] <= next_ninq:
                            next_ninq = next_table[dst, network_topology[next_node][i]]
                if Q_or_CQ == 'Q':
                    q_est = qtable[dst, next_node] + learn_rate * (packet[3] + 1 + next_ninq -  qtable[dst, next_node])
                else:
                    if rule3a_or_not == 0:
                        c_est = lamda * ctable[dst, next_node]
                    else:
                        c_est = ctable[dst, next_node] + max(ctable[dst, next_node],-packet[3]-next_ninq) * (packet[3] + 1 + next_ninq - ctable[dst, next_node])
                    if c_est >1:
                        c_est = 0.2
                    coef = max(ctable[dst, next_node],1-c_est)
                    q_est = qtable[dst, next_node] + coef * (packet[3] + 1 + next_ninq-qtable[dst, next_node])
                    if q_est == qtable[dst, next_node]:
                        rule3a_or_not = 0
                    else:
                        rule3a_or_not = 1
                    C_Table[pnode][dst, next_node] = c_est
                Q_Table[pnode][dst, next_node] = q_est
                [packet_list[packet_index][2], packet_list[packet_index][3], packet_list[packet_index][4]] = \
                    [next_node, 0, packet_list[packet_index][4] + 1]
                if next_node != dst:
                    node_list[next_node].put(packet_index)
                else:
                    packet_list[packet_index][5] = 1
                    activenode_list.remove(packet_index)
                    avg_time = avg_time + packet_list[packet_index][4]
                    avg_nodes = avg_nodes + 1
        time_count = time_count +  1
        avg_count = avg_count + 1
        if avg_count == 200:
            avg_count = 0
            avg_list.append(avg_time / avg_nodes)
    if Q_or_CQ == 'Q':
        # avg_list_q = avg_list
        x = len(avg_list)
        X = list()
        for x_i in range(x):
            X.append(x_i*200)
        plt.plot(X, avg_list,'b-')
    else:
        x = len(avg_list)
        X = list()
        for x_i in range(x):
            X.append(x_i*200)
        plt.plot(X, avg_list,'r--')
        plt.xlabel('Steps')
        plt.ylabel('Average Delivery Time')
        # plt.show()
        plt.savefig('CQ.png')

if __name__ == "__main__":
    # load = 2.15 # Medium
    load = 2.75 # High
    Experiments_Q_CQ_Routing(load = load,max_steps= 15000, Q_or_CQ='Q', learn_rate=0.85, lamda=0.95)
    Experiments_Q_CQ_Routing(load = load,max_steps= 15000, Q_or_CQ='CQ', learn_rate=0.85, lamda=0.95)