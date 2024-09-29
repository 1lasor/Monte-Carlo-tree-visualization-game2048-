import os
import time
import torch
import pickle
import random
import numpy as np
from math import log2
import matplotlib.pyplot as plt
# from Net import NetLowReward
# from ex_pool import ReplayBuffer
# from game.gym2048.Game2048Env import Game2048Env, unstack, stack
from Game2048Env import Game2048Env

from pyecharts import options as opts
from pyecharts.charts import Graph



device = torch.device("cuda")
# Define actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# echarts配置
direction = ['上','右','下','左']
all_nodes = []
all_links = []

def stack(flat, layers=16):  # Y original layers=16
    # if flag_Y:
    #     layers=1
    """Convert an [4, 4] representation into [4, 4, layers] with one layers for each value."""
    # representation is what each layer represents
    representation = 2 ** (np.arange(layers, dtype=int) + 1)

    # layered is the flat board repeated layers times
    layered = np.repeat(flat[:, :, np.newaxis], layers, axis=-1)

    # Now set the values in the board to 1 or zero depending whether they match representation.
    # Representation is broadcast across a number of axes
    layered = np.where(layered == representation, 1, 0)

    return layered


def unstack(layered):
    """Convert an [4, 4, layers] representation back to [4, 4]."""
    # representation is what each layer represents
    layers = layered.shape[-1]
    representation = 2 ** (np.arange(layers, dtype=int) + 1)

    # Use the representation to convert binary layers back to original values
    original = np.zeros((4, 4), dtype=int)
    for i in range(layers):
        # Convert the result to integer before adding
        addition = (layered[:, :, i] * representation[i]).astype(int)
        original += addition

    return original


# original
class TreeNode:
    def __init__(self, parent, action, value=1, visits=1):
        self.parent = parent
        self.action = action
        self.children = []
        self.value = value
        self.visits = visits
        self.id = 0
        self.s = []

    def add_child(self, action, value=1, visits=1):
        child = TreeNode(self, action, value, visits)
        self.children.append(child)
        return child

    def is_root(self):
        """判断节点是否为根节点"""
        return self.parent is None

    def print_info(self):
        print("Parent:", self.parent)
        print("Action:", self.action)
        print("Children:", [child.action for child in self.children])
        print("Value:", self.value)
        print("Visits:", self.visits)

    # def to_dot(self, dot=None):
    #     if dot is None:
    #         dot = graphviz.Digraph()
    #
    #     color = 'black'
    #     if self.value >= 30:
    #         color = 'red'
    #     elif self.value >= 20 and self.value < 30:
    #         color = 'orange'
    #     elif self.value >= 10 and self.value < 20:
    #         color = 'yellow'
    #
    #     edge_color = 'black'
    #     if self.visits >=100:
    #         edge_color = 'red'
    #
    #
    #     dot.node(str(id(self)), fontcolor='white',color=color,style='filled', label=f"{str(self.action)}--{self.visits}s\n({self.value:.2f})")
    #
    #     for child in self.children:
    #         child.to_dot(dot)
    #         dot.edge(str(id(self)), str(id(child)),color = edge_color)
    #
    #     return dot

    def count_nodes(self):
        count = 1  # 当前节点
        for child in self.children:
            count += child.count_nodes()  # 递归统计子节点数
        return count

    def to_echart(self):

        # 根据当前状态创建图片
        fig, ax = plt.subplots()
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        for i in range(4):
            for j in range(4):
                value = self.s[i, j]
                #
                if value == 0:
                    color = 'black'
                else:
                    color = plt.cm.plasma(log2(value) / 11)
                ax.add_patch(plt.Rectangle((j, 3 - i), 1, 1, edgecolor='black', facecolor=color))
                # 在矩形中添加文本
                ax.text(j + 0.5, 3 - i + 0.5, str(value), ha='center', va='center', fontsize=24,color='white')
        ax.axis('off')

        # 保存图片
        self.id = len(all_nodes)
        plt.savefig(f'./images/{self.id}.png')
        plt.close()

        all_nodes.append({'name':self.id,'symbol':f'image://images/{self.id}.png','value':f"节点价值: {self.value:.2f}\n访问次数: {self.visits}"})
        for child in self.children:
            child.to_echart()
            all_links.append(opts.GraphLink(source=f"{self.id}", target=f"{child.id}",value=child.action,label_opts=opts.LabelOpts(
                is_show=True, position="middle", formatter=f"{self.id} 的动作 {direction[child.action]}",font_size=10
            )))



class MonteCarloTreeSearch:
    def __init__(self, env):
        self.root = TreeNode(None, None)
        self.max_depth = 80
        self.env = env
        self.action_history = []

    def select_action(self, state, l=16):
        c = 0
        count = 0
        t0 = time.time()
        s0 = unstack(state.copy())
        self.root.s = unstack(state.copy())
        for i in range(l):  # Perform 100 MCTS iterations
            node = self.root
            s = unstack(state.copy())  # 第一次根结点需要赋值
            self.env.reset_to_state(s)  # 重置游戏初始环境

            # Selection
            chi = 0
            # 用于记录遍历过程中经过的节点
            traversed_nodes = []
            while node.children:
                # Check if the current node is non-full
                if len(node.children) < len(self.get_valid_actions(s)):
                    break
                if node.is_root():
                    traversed_nodes.append(node)

                selected_index = self.uct(node)
                node = node.children[selected_index]
                a = node.action
                s_, r, _, _, _, _, end = self.env.step(a)
                traversed_nodes.append(node)  # 记录当前节点
                s = unstack(s_)


            # node.print_info()

            # Expansion
            possible_actions = self.get_valid_actions(s)

            available_actions = [action for action in possible_actions if
                                 action not in [child.action for child in node.children]]

            if available_actions:
                expansion_action = random.choice(available_actions)
                node = node.add_child(expansion_action)
                s_, r, _, _, _, _, end = self.env.step(expansion_action)
                s = unstack(s_)
                traversed_nodes.append(node)  # 记录扩展结点
                node.s = s
                # 模拟
                sim_state = s.copy()
                depth = 0
                r = 0
                while not self.is_terminal(sim_state) and depth < self.max_depth:
                    sim_action = random.choice(self.get_valid_actions(sim_state))
                    sim_state, reward, _, _, _, _, _ = self.env.step(sim_action)
                    sim_state = unstack(sim_state)
                    r = reward + r
                    depth += 1
                # print(sim_state,depth)

                # 回溯更新
                for n in traversed_nodes:
                    n.visits += 1
                    n.value = (n.visits / (n.visits + 1)) * n.value + (1 / (n.visits + 1)) * r
            else:  # 说明此时的游戏已经结束

                c = c + 1
                for n in traversed_nodes:
                    n.visits += 1
                    n.value = (n.visits / (n.visits + 1)) * n.value

            # print('第',i,'次搜索结束')
            action_index = self.uct(self.root)
            action = self.root.children[action_index].action
            self.action_history.append(action)

        # 选择根结点下的最优子结点（动作）
        best_action_index = self.uct(self.root)
        best_action = self.root.children[best_action_index].action
        # print(count)
        # print(time.time()-t0)
        total_nodes = self.root.count_nodes()
        # print("Total nodes in the tree:", total_nodes)
        # print('游戏结束状态个数',c)
        self.env.reset_to_state(s0)
        return best_action

    def get_action_history(self):
        return self.action_history

    def uct(self, node):
        uct_values = []
        for child in node.children:
            if child.visits == 0:
                uct_values.append(float('inf'))
            else:
                uct = child.value / node.value + 0.8 * np.sqrt(np.log(node.visits) / child.visits)
                uct_values.append(uct)
        return np.argmax(uct_values)

    def get_valid_actions(self, state):
        valid_actions = []
        for action in [UP, DOWN, LEFT, RIGHT]:
            if self.is_valid_action(state, action):
                valid_actions.append(action)
                self.env.reset_to_state(state)
        return valid_actions

    def is_valid_action(self, state, action):
        # Make a copy of the current state
        s = state.copy()

        # Perform the action on the new state
        new_state = self.take_action(s, action)

        # Check if the new state is different from the original state
        return not np.array_equal(state, new_state)

    def take_action(self, state, action):
        self.env.reset_to_state(state)
        # Perform action on given state and return new state
        s_, r, _, _, _, _, _ = self.env.step(action)
        return unstack(s_)

    def is_terminal(self, state):
        # Check if state is terminal (game over)
        return self.env.isend()

    def calculate_tree_height(self):
        stack = [(self.root, 0)]
        max_height = 0
        while stack:
            node, height = stack.pop()
            max_height = max(max_height, height)
            for child in node.children:
                stack.append((child, height + 1))
        self.tree_height = max_height

    def explore_root(self,state):
        node = self.root
        s = unstack(state.copy())
        possible_actions = self.get_valid_actions(s)
        available_actions = [action for action in possible_actions]

        while available_actions:
            expansion_action = random.choice(available_actions)
            node = node.add_child(expansion_action)
            s_, r, _, _, _, _, end = self.env.step(expansion_action)
            s = unstack(s_)
            # traversed_nodes.append(node)  # 记录扩展结点

            # 模拟
            sim_state = s.copy()
            depth = 0
            r = 0
            while not self.is_terminal(sim_state) and depth < self.max_depth:
                sim_action = random.choice(self.get_valid_actions(sim_state))
                sim_state, reward, _, _, _, _, _ = self.env.step(sim_action)
                sim_state = unstack(sim_state)
                r = reward + r
                depth += 1
            # print(sim_state,depth)
        else:  # 说明此时的游戏已经结束
            pass
# adapted
# class TreeNode:
#     def __init__(self, parent, action, value=1, value_sum=1, visits=1,possibility=0):
#         self.parent = parent
#         self.action = action
#         self.children = []
#         self.value = value
#         self.value_sum = value_sum
#         self.visits = visits
#         self.possibility = possibility
#         self.o_next = []
#
#     def add_child(self, action, value=1, visits=1):
#         child = TreeNode(self, action, value, visits)
#         self.children.append(child)
#         return child
#
#     def is_root(self):
#         """判断节点是否为根节点"""
#         return self.parent is None
#
#     def print_info(self):
#         print("Parent:", self.parent)
#         print("Action:", self.action)
#         print("Children:", [child.action for child in self.children])
#         print("Value:", self.value)
#         print("Visits:", self.visits)
#
#     def to_dot(self, dot=None):
#         if dot is None:
#             dot = graphviz.Digraph()
#
#         dot.node(str(id(self)), label=str(self.action))
#
#         for child in self.children:
#             child.to_dot(dot)
#             dot.edge(str(id(self)), str(id(child)))
#
#         return dot
#
#     def count_nodes(self):
#         count = 1  # 当前节点
#         for child in self.children:
#             count += child.count_nodes()  # 递归统计子节点数
#         return count
#
#
# class MonteCarloTreeSearch:
#     def __init__(self, env):
#         self.root = TreeNode(None, None)
#         self.max_depth = 80
#         self.env = env
#         self.action_history = []
#
#     def select_action(self, state, l=16, model=None,rate=0.99):
#         device = torch.device("cuda")
#         # print("selecting action")
#         c = 0
#         count = 0
#         t0 = time.time()
#         s0 = unstack(state.copy())
#         for i in range(l):  # Perform 100 MCTS iterations
#             node = self.root
#             s = unstack(state.copy())  # 第一次根结点需要赋值
#             state_now = state.copy()
#             self.env.reset_to_state(s)  # 重置游戏初始环境
#
#             # Selection
#             chi = 0
#             # 用于记录遍历过程中经过的节点
#             traversed_nodes = []
#             traversed_rewards = []
#
#             #找到叶子节点
#             while node.children:
#                 # Check if the current node is non-full
#                 # 如果当前节点分支数小于可执行动作数
#                 if len(node.children) < len(self.get_valid_actions(s)):
#                     break
#                 if node.is_root():
#                     traversed_nodes.append(node)
#
#                 #用uct选择路径
#                 selected_index = self.uct(node)
#                 node = node.children[selected_index]
#
#                 a = node.action
#                 #print("nodeaction",a)
#                 s_, r, _, _, _, _, end = self.env.step(a)
#                 traversed_nodes.append(node)  # 记录当前节点
#                 traversed_rewards.append(r)
#                 state_now = s_
#                 s = unstack(s_)
#
#             #此时node为叶子节点
#             # node.print_info()
#
#             # Expansion
#             possible_actions = self.get_valid_actions(s)
#             available_actions = [action for action in possible_actions if
#                                  action not in [child.action for child in node.children]]
#             #print(available_actions)
#             if available_actions:
#                 expansion_action = random.choice(available_actions)
#                 node = node.add_child(expansion_action)
#                 #print("expansion_action", expansion_action)
#                 s_, r, _, _, _, _, end = self.env.step(expansion_action)
#                 s = unstack(s_)
#                 #print(s)
#                 traversed_nodes.append(node)  # 记录扩展结点
#                 traversed_rewards.append(r)
#
#                 # 模拟
#                 if model is not None:
#                     #print("模拟")
#                     sim_state = state_now
#                     s_tensor = torch.FloatTensor(np.expand_dims(sim_state, axis=0)).to(device)
#                     #print(s_tensor)
#                     with torch.no_grad():
#                         q = model(s_tensor)
#                         q_eval = q[0,expansion_action].cpu().numpy()
#                         #print(q_eval)
#
#                 else:
#                     sim_state = s.copy()
#                     depth = 0
#                     r = 0
#                     while not self.is_terminal(sim_state) and depth < self.max_depth:
#                         sim_action = random.choice(self.get_valid_actions(sim_state))
#                         sim_state, reward, _, _, _, _, _ = self.env.step(sim_action)
#                         sim_state = unstack(sim_state)
#                         r = reward + r
#                         depth += 1
#
#                 # 回溯更新
#                 value_tmp = q_eval
#                 #print(value_tmp)
#                 for n, reward in zip(traversed_nodes[:-1][::-1], traversed_rewards[::-1]):
#                     value_tmp = reward + value_tmp * rate
#                     n.visits += 1
#                     n.value = (n.visits / (n.visits + 1)) * n.value + (1 / (n.visits + 1)) * value_tmp
#             else:  # 说明此时的游戏已经结束
#
#                 c = c + 1
#                 for n, reward in zip(traversed_nodes[:-1][::-1], traversed_rewards[::-1]):
#                     n.visits += 1
#                     n.value = (n.visits / (n.visits + 1)) * n.value
#
#             # print('第',i,'次搜索结束')
#             action_index = self.uct(self.root)
#             action = self.root.children[action_index].action
#             #print(action)
#             self.action_history.append(action)
#
#         # 选择根结点下的最优子结点（动作）
#         best_action_index = self.uct(self.root)
#         best_action = self.root.children[best_action_index].action
#         self.env.reset_to_state(s0)
#         # print(best_action)
#         return best_action
#
#     def get_action_history(self):
#         return self.action_history
#
#     def uct(self, node):
#         uct_values = []
#         for child in node.children:
#             if child.visits == 0:
#                 uct_values.append(float('inf'))
#             else:
#                 uct = child.value / node.value + 0.8 * np.sqrt(np.log(node.visits) / child.visits)
#                 uct_values.append(uct)
#         return np.argmax(uct_values)
#
#     def get_valid_actions(self, state):
#         valid_actions = []
#         for action in [UP, DOWN, LEFT, RIGHT]:
#             if self.is_valid_action(state, action):
#                 valid_actions.append(action)
#                 self.env.reset_to_state(state)
#         return valid_actions
#
#     def is_valid_action(self, state, action):
#         # Make a copy of the current state
#         s = state.copy()
#
#         # Perform the action on the new state
#         new_state = self.take_action(s, action)
#
#         # Check if the new state is different from the original state
#         return not np.array_equal(state, new_state)
#
#     def take_action(self, state, action):
#         self.env.reset_to_state(state)
#         # Perform action on given state and return new state
#         s_, r, _, _, _, _, _ = self.env.step(action)
#         return unstack(s_)
#
#     def is_terminal(self, state):
#         # Check if state is terminal (game over)
#         return self.env.isend()
#
#     def calculate_tree_height(self):
#         stack = [(self.root, 0)]
#         max_height = 0
#         while stack:
#             node, height = stack.pop()
#             max_height = max(max_height, height)
#             for child in node.children:
#                 stack.append((child, height + 1))
#         self.tree_height = max_height

def mcts(state):
    env = Game2048Env()
    state = np.array(state)

    # mcts可视化
    mcts = MonteCarloTreeSearch(env)
    action = mcts.select_action(stack(state), 50)
    mcts.root.to_echart()  # 使用根节点的to_dot方法来创建图形

    html_file_name = "mcts_graph_with_edge_options.html"

    c = (

        Graph(init_opts=opts.InitOpts(width="100%", height="960px"))
        .add(
            "",
            all_nodes,
            all_links,
            repulsion=4000,
            # edge_label=opts.LabelOpts(
            #     is_show=True, position="middle", formatter="{b} 的动作 {}",font_size=10
            # ),
            symbol_size=100
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="2048蒙特卡洛树搜索数据可视化"),
        )

        .render(html_file_name)

    )

    all_nodes.clear()
    all_links.clear()
    os.system(f"start {html_file_name}")

if __name__ == "__main__":
    env = Game2048Env()
    # model_path = "rollout.pkl"
    # with open(model_path, 'rb') as file:
    #     model_data = pickle.load(file)
    #
    # model = model_data['model'].to(device)

    # # 测试模块
    # count = 100
    # win_count = 0
    #
    # for i in range(count):
    #     print(f"Ep: {i}")
    #     s = env.reset()
    #     while True:
    #         # print(unstack(s))
    #         mcts = MonteCarloTreeSearch(env)
    #         action = mcts.select_action(s.copy(),200,model)
    #
    #         # action = torch.tensor(torch.argmax(model(torch.FloatTensor(np.expand_dims(s, axis=0)).to(device)), 1).item())
    #
    #         # print(f"action:{action}\n")
    #         sim_state, reward, done, _, _, _, done_end = env.step(action)
    #
    #         if done or env.highest()>=1024:
    #             print(f"The end:\n{unstack(s)}\ndue to {action}\n")
    #             if env.highest()>=1024:
    #                 win_count+=1
    #             break
    #
    #         s = sim_state
    #
    # success_rate = win_count/count
    # print(f"The success rate of mcts is {success_rate}")
    # 全局胜率：修改过的mct（200）：0.24

    # 单步测试模块
    use_dqn_after_key_action = False
    state = np.array([[16, 8, 4, 2], [2, 256, 8, 32], [512, 16, 2, 0], [8, 64, 2, 0]])
    # 关键节点胜率： 修改过的mct（500）：0.5-0.8，修改过的mct（200）：0.9，修改过的mct（100）：0.4,dqn：0.2-0.3

    # mcts可视化测试
    mcts = MonteCarloTreeSearch(env)
    action = mcts.select_action(stack(state), 50)
    mcts.root.to_echart()  # 使用根节点的to_dot方法来创建图形

    html_file_name = "mcts_graph_with_edge_options.html"

    c = (

        Graph(init_opts=opts.InitOpts(width="100%", height="960px"))
        .add(
            "",
            all_nodes,
            all_links,
            repulsion=4000,
            # edge_label=opts.LabelOpts(
            #     is_show=True, position="middle", formatter="{b} 的动作 {}",font_size=10
            # ),
            symbol_size = 100
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="2048蒙特卡洛树搜索数据可视化"),
        )

        .render(html_file_name)

    )

    all_nodes.clear()
    all_links.clear()
    os.system(f"start {html_file_name}")

    # # 动作稳定性测试
    # action_list = []
    # for i in range(100):
    #     test_state = stack(state)
    #     mcts = MonteCarloTreeSearch(env)
    #     action = mcts.select_action(test_state, 500,model)
    #     action_list.append(action)
    #     print(f"test_state:\n{unstack(test_state)}\naction:{action}")
    #
    # [print(f"{x}:{action_list.count(x)}") for x in [0,1,2,3]]
    # action_count_list = [action_list.count(x) for x in [0,1,2,3]]
    # stability = max(action_count_list)/len(action_list)
    # print(f"stability: {stability}")
    # # 动作稳定性: 修改过的mct（500）:0.67（2）,修改过的mct(加概率权重）（500）:0.63（2）,未修改过的mct（500）:0.5（0）,

    # # 关键节点胜率测试
    # test_rounds = 10  # 总局数
    # win_rounds = 0
    #
    # test_state = stack(state)
    # mcts = MonteCarloTreeSearch(env)
    # key_action = mcts.select_action(test_state, 500,model)
    # # print("testing")
    # for test in range(test_rounds):
    #     print(f"Ep: {test}|key action: {key_action}")
    #     # 重置环境
    #     env.reset_to_state(state.copy())
    #     key_state, key_reward, key_done, _, _, _, key_done_end = env.step(key_action)
    #     test_state = key_state
    #     while True:
    #         # print(unstack(test_state))
    #
    #         if use_dqn_after_key_action:
    #             s_tensor = torch.FloatTensor(np.expand_dims(test_state, axis=0)).to(device)
    #
    #             with torch.no_grad():
    #                 q = model(s_tensor)
    #                 action = torch.argmax(q, 1).item()
    #         else:
    #             mcts = MonteCarloTreeSearch(env)
    #             action = mcts.select_action(test_state, 200, model)
    #         # print(f"action:{action}\n")
    #
    #         sim_state, reward, done, _, _, _, done_end = env.step(action)
    #         test_state = sim_state
    #         # 如果卡死或者到达指定分数游戏结束
    #         if  done or env.highest() >= 1024:
    #             if env.highest() >= 1024:
    #                 win_rounds += 1
    #             break
    #
    # print(f"success rate:{win_rounds/test_rounds}")
