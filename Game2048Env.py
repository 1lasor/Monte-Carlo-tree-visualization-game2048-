import sys
import logging
import time

from six import StringIO
import itertools
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


Y = True

class IllegalMove(Exception):
    pass


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def stack(flat, layers=16):
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


class Game2048Env(gym.Env):
    metadata = {'render.modes': ['ansi', 'human', 'rgb_array']}

    def __init__(self):
        # Definitions for game. Board must be square.
        self.size = 4
        self.w = self.size
        self.h = self.size
        self.squares = self.size * self.size

        # Maintain own idea of game score, separate from rewards
        self.score = 0

        # Members for gym implementation
        self.action_space = spaces.Discrete(4)
        # Suppose that the maximum tile is as if you have powers of 2 across the board.
        layers = self.squares
        self.observation_space = spaces.Box(0, 1, (self.w, self.h, layers), int)
        self.set_illegal_move_reward(-20)
        self.set_max_tile(None)

        # Size of square for rendering
        self.grid_size = 70

        # Initialise seed
        self.seed()

        # Reset ready for a game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_illegal_move_reward(self, reward):
        """Define the reward/penalty for performing an illegal move. Also need
            to update the reward range for this."""
        # Guess that the maximum reward is also 2**squares though you'll probably never get that.
        # (assume that illegal move reward is the lowest value that can be returned
        self.illegal_move_reward = reward
        self.reward_range = (self.illegal_move_reward, float(2 ** self.squares))

    def set_max_tile(self, max_tile):
        """Define the maximum tile that will end the game (e.g. 2048). None means no limit.
           This does not affect the state returned."""
        assert max_tile is None or isinstance(max_tile, int)
        self.max_tile = max_tile

    # Implement gym interface
    def step(self, action):
        """Perform one step of the game. This involves moving and adding a new tile."""
        logging.debug("Action {}".format(action))
        score = 0
        done = None
        done_end = None
        info = {
            'illegal_move': False,
        }
        try:
            score = float(self.move(action))
            if score > 0:
                score = score
            if score < 0:
                score = 0
            self.score += score
            assert score <= 2 ** (self.w * self.h)
            p = self.add_tile()
            done = self.isend()
            done_end = self.isend()
            if Y:
                reward = float(score)*p
            else:
                reward = float(score)

        except IllegalMove:
            logging.debug("Illegal move")
            info['illegal_move'] = True
            done = True
            done_end = self.isend()
            reward = self.illegal_move_reward

        # Calculate the number of empty spaces
        num_empty_spaces = len(self.empties())
        info['highest'] = self.highest()
        ac = self.Matrix
        # Return observation (board state), reward, done, info dict, and number of empty spaces
        return stack(self.Matrix), reward, done, info, num_empty_spaces, ac, done_end

    def all_step(self, action):
        info = {
            'illegal_move': False,
        }
        states = []
        try:
            score = float(self.move(action))
            if score > 0:
                score = score
            if score < 0:
                score = 0
            self.score += score
            assert score <= 2 ** (self.w * self.h)

            empties = self.empties()
            assert empties.shape[0]
            for possible_tiles in [2, 4]:
                val = possible_tiles
                for empty_idx in empties:
                    tmp_state = self.Matrix.copy()
                    tmp_state[empty_idx[0],empty_idx[1]] = val
                    states.append(tmp_state)

            #return 1 / empties.size * (0.9 if val == 2 else 0.1)

            return states

        except IllegalMove:
            logging.debug("Illegal move")
            info['illegal_move'] = True
            done = True
            done_end = self.isend()
            reward = self.illegal_move_reward

    def reset(self):
        self.Matrix = np.zeros((self.h, self.w), int)
        self.score = 0
        logging.debug("Adding tiles")
        self.add_tile()
        self.add_tile()

        return stack(self.Matrix)

    def reset_to_state(self, s):
        # 将棋局重置为给定的状态 s
        self.Matrix = s
        self.score = 0  # 重置分数，你可能需要根据 s 中的信息重新计算分数
        # 返回重置后的棋局
        return self.Matrix

    def render(self, mode='human'):
        if mode == 'rgb_array':
            black = (0, 0, 0)
            grey = (200, 200, 200)
            white = (255, 255, 255)
            tile_colour_map = {
                2: (255, 255, 255),
                4: (255, 248, 220),
                8: (255, 222, 173),
                16: (244, 164, 96),
                32: (205, 92, 92),
                64: (240, 255, 255),
                128: (240, 255, 240),
                256: (193, 255, 193),
                512: (154, 255, 154),
                1024: (84, 139, 84),
                2048: (139, 69, 19),
                4096: (178, 34, 34),
            }
            grid_size = self.grid_size

            # Render with Pillow
            pil_board = Image.new("RGB", (grid_size * 4, grid_size * 4))
            draw = ImageDraw.Draw(pil_board)
            draw.rectangle([0, 0, 4 * grid_size, 4 * grid_size], grey)
            fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)

            for y in range(4):
                for x in range(4):
                    o = self.get(y, x)
                    if o:
                        draw.rectangle([x * grid_size, y * grid_size, (x + 1) * grid_size, (y + 1) * grid_size],
                                       tile_colour_map[o])
                        # 使用textbbox方法来获取文本的包围盒
                        text_bbox = draw.textbbox((x * grid_size, y * grid_size), str(o), font=fnt)
                        text_x_size = text_bbox[2] - text_bbox[0]
                        text_y_size = text_bbox[3] - text_bbox[1]
                        draw.text(((x * grid_size + (grid_size - text_x_size) // 2),
                                   (y * grid_size + (grid_size - text_y_size) // 2)), str(o), font=fnt, fill=black)
                        assert text_x_size < grid_size
                        assert text_y_size < grid_size

            return np.asarray(pil_board)

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        s = 'Score: {}\n'.format(self.score)
        s += 'Highest: {}\n'.format(self.highest())
        npa = np.array(self.Matrix)
        grid = npa.reshape((self.size, self.size))
        s += "{}\n".format(grid)
        outfile.write(s)
        return outfile

    # Implement 2048 game
    def add_tile(self):
        possible_tiles = np.array([2, 4])
        tile_probabilities = np.array([0.9, 0.1])
        val = self.np_random.choice(possible_tiles, 1, p=tile_probabilities)[0]
        empties = self.empties()
        assert empties.shape[0]
        empty_idx = self.np_random.choice(empties.shape[0])
        empty = empties[empty_idx]
        logging.debug("Adding %s at %s", val, (empty[0], empty[1]))
        self.set(empty[0], empty[1], val)
        return 1/empties.size * (0.9 if val==2 else 0.1)


    def get(self, x, y):
        """Return the value of one square."""
        return self.Matrix[x, y]

    def set(self, x, y, val):
        """Set the value of one square."""
        self.Matrix[x, y] = val

    def empties(self):
        """Return a 2d numpy array with the location of empty squares."""
        return np.argwhere(self.Matrix == 0)

    def highest(self):
        """Report the highest tile on the board."""
        return np.max(self.Matrix)

    def board_total(self):
        """Calculate the total value of all tiles on the board."""
        return np.sum(self.Matrix)

    # def move(self, direction, trial=False):
    #     """Perform one move of the game. Shift things to one side then,
    #     combine. directions 0, 1, 2, 3 are up, right, down, left.
    #     Returns the maximum score that [would have] got from the move."""
    #     if not trial:
    #         if direction == 0:
    #             logging.debug("Up")
    #         elif direction == 1:
    #             logging.debug("Right")
    #         elif direction == 2:
    #             logging.debug("Down")
    #         elif direction == 3:
    #             logging.debug("Left")
    #
    #     changed = False
    #     scores = []  # 修改为列表，用于存储每次移动得到的分数
    #     dir_div_two = int(direction / 2)
    #     dir_mod_two = int(direction % 2)
    #     shift_direction = dir_mod_two ^ dir_div_two  # 0 for towards up left, 1 for towards bottom right
    #
    #     # Construct a range for extracting row/column into a list
    #     rx = list(range(self.w))
    #     ry = list(range(self.h))
    #
    #
    #     if dir_mod_two == 0:
    #         # Up or down, split into columns
    #         # print(direction)
    #         for y in range(self.h):
    #             old = [self.get(x, y) for x in rx]
    #
    #             # print('旧',old)
    #             (new, ms) = self.shift(old, shift_direction)
    #             # print('新',new)
    #             scores.append(ms)  # 添加到分数列表中
    #             if old != new:
    #                 changed = True
    #                 if not trial:
    #                     for x in rx:
    #                         self.set(x, y, new[x])
    #     else:
    #         # print(direction)
    #         # Left or right, split into rows
    #         for x in range(self.w):
    #             old = [self.get(x, y) for y in ry]
    #             # print('旧', old)
    #             (new, ms) = self.shift(old, shift_direction)
    #             # print('新', new)
    #             scores.append(ms)  # 添加到分数列表中
    #             if old != new:
    #                 changed = True
    #                 if not trial:
    #                     for y in ry:
    #                         self.set(x, y, new[y])
    #         # time.sleep(100)
    #
    #     if not changed:
    #         raise IllegalMove
    #         # 打印分数列表和最大分数
    #
    #     # 获取列表中的最大值作为 move_scores 返回
    #
    #     move_scores = max(scores) if scores else 0
    #     # print("Scores from this move:", scores)
    #     # print("Maximum score from this move:", move_scores)
    #
    #
    #     return move_scores
    def move(self, direction, trial=False):
        """Perform one move of the game. Shift things to one side then,
        combine. directions 0, 1, 2, 3 are up, right, down, left.
        Returns the sum of scores obtained from each move."""
        if not trial:
            if direction == 0:
                logging.debug("Up")
            elif direction == 1:
                logging.debug("Right")
            elif direction == 2:
                logging.debug("Down")
            elif direction == 3:
                logging.debug("Left")

        changed = False
        scores = []  # 修改为列表，用于存储每次移动得到的分数
        dir_div_two = int(direction / 2)
        dir_mod_two = int(direction % 2)
        shift_direction = dir_mod_two ^ dir_div_two  # 0 for towards up left, 1 for towards bottom right

        # Construct a range for extracting row/column into a list
        rx = list(range(self.w))
        ry = list(range(self.h))

        if dir_mod_two == 0:
            # Up or down, split into columns
            for y in range(self.h):
                old = [self.get(x, y) for x in rx]
                (new, ms) = self.shift(old, shift_direction)
                scores.append(ms)  # 添加到分数列表中
                if old != new:
                    changed = True
                    if not trial:
                        for x in rx:
                            self.set(x, y, new[x])
        else:
            # Left or right, split into rows
            for x in range(self.w):
                old = [self.get(x, y) for y in ry]
                (new, ms) = self.shift(old, shift_direction)
                scores.append(ms)  # 添加到分数列表中
                if old != new:
                    changed = True
                    if not trial:
                        for y in ry:
                            self.set(x, y, new[y])

        if not changed:
            raise IllegalMove

        # 返回移动得分列表的总和
        return sum(scores)

    def combine(self, shifted_row):
        """Combine same tiles when moving to one side. This function always
           shifts towards the left. Also count the score of combined tiles."""
        move_score = 0
        combined_row = [0] * self.size
        skip = False
        output_index = 0
        for p in pairwise(shifted_row):
            if skip:
                skip = False
                continue
            combined_row[output_index] = p[0]
            if p[0] == p[1]:
                combined_row[output_index] += p[1]
                move_score += p[0] + p[1]
                # Skip the next thing in the list.
                skip = True
            output_index += 1
        if shifted_row and not skip:
            combined_row[output_index] = shifted_row[-1]

        return (combined_row, move_score)

    def shift(self, row, direction):
        """Shift one row left (direction == 0) or right (direction == 1), combining if required."""
        length = len(row)
        assert length == self.size
        assert direction == 0 or direction == 1

        # Shift all non-zero digits up
        shifted_row = [i for i in row if i != 0]

        # Reverse list to handle shifting to the right
        if direction:
            shifted_row.reverse()

        (combined_row, move_score) = self.combine(shifted_row)

        # Reverse list to handle shifting to the right
        if direction:
            combined_row.reverse()

        assert len(combined_row) == self.size
        return (combined_row, move_score)

    def isend(self):
        """Has the game ended. Game ends if there is a tile equal to the limit
           or there are no legal moves. If there are empty spaces then there
           must be legal moves."""

        # if self.max_tile is not None and self.highest() == self.max_tile:
        #     return True

        for direction in range(4):
            try:
                self.move(direction, trial=True)
                # Not the end if we can do any move
                return False
            except IllegalMove:
                pass
        return True

    def get_board(self):
        """Retrieve the whole board, useful for testing."""
        return self.Matrix

    def set_board(self, new_board):
        """Retrieve the whole board, useful for testing."""
        self.Matrix = new_board
