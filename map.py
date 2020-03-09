import random, time


class Game:
    def __init__(self):
        random.seed(time.time())
        self.gas = 250
        self.money = 1000
        self.gas_max = 500
        self.map = []
        self.map_size = 15
        self.player_pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
        self.is_done = False
        self.reward = 0
        self.prepaid = 0.1
        self.quests = [[12, 6], [0, 2], [11, 11], [5, 4], [4, 13]]
        self.destinations = [[6, 7], [2, 5], [13, 2], [5, 10], [2, 2]]
        self.rewards = [300, 200, 400, 250, 500]
        self.gas_price = 1
        self.gas_stations = [[7, 7], [0, 5], [10, 6]]
        self.has_tanked = False
        self.started = False
        self.ended = False

        self.cargo = [0, 0, 0, 0, 0]
        self.action_space = (0, 1, 2, 3, 4)
        self.action_count = len(self.action_space)
        self.state_count = self.map_size * 2 + 1 + len(self.quests)
        self.observation_space = []
        self.actions = [self.action_up, self.action_down, self.action_left, self.action_right, self.action_special]
        # self.actions = [self.action_up, self.action_down, self.action_left, self.action_right, self.action_wait, self.action_special]

        for i in range(self.map_size):
            temp = []
            for y in range(self.map_size):
                temp.append("XX")
            self.map.append(temp)

        self.map[0][0] = "ST"

        for i in range(len(self.quests)):
            quest = self.quests[i]
            self.map[quest[0]][quest[1]] = "Q" + str(i + 1)

        for i in range(len(self.destinations)):
            destination = self.destinations[i]
            self.map[destination[0]][destination[1]] = "R" + str(i + 1)

        for gas_station in self.gas_stations:
            self.map[gas_station[0]][gas_station[1]] = "LP"

    def reset(self):
        self.player_pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
        self.gas = 250
        self.money = 1000
        self.gas_max = 500
        self.cargo = [0, 0, 0, 0, 0]
        self.is_done = False
        self.reward = 0
        self.has_tanked = False
        self.started = False
        self.ended = False

        return self.get_state_object()

    def get_state_object(self):
        state = (self.player_pos, self.cargo, self.gas)
        return state

    def step(self, action):
        if action > len(self.action_space) - 1:
            raise Exception('InvalidAction', action)
        else:
            reward = self.actions[action]()
            if self.gas <= 0:
                reward -= 5000
                self.is_done = True

        state = (self.get_state_object(), reward, self.is_done, [])
        return state

    def action_up(self):
        if self.player_pos[0] == 0:
            return self.action_wait()
        else:
            self.player_pos[0] -= 1
            self.gas -= 1
            return 0

    def action_down(self):
        if self.player_pos[0] == 14:
            return self.action_wait()
        else:
            self.gas -= 1
            self.player_pos[0] += 1
            return 0

    def action_right(self):
        if self.player_pos[1] == 14:
            return self.action_wait()
        else:
            self.gas -= 1
            self.player_pos[1] += 1
            return 0

    def action_left(self):
        if self.player_pos[1] == 0:
            return self.action_wait()
        else:
            self.gas -= 1
            self.player_pos[1] -= 0
            return 0

    def action_wait(self):
        self.gas -= 0.25
        return 0

    def action_special(self):
        if self.player_pos in self.quests:
            quest = self.quests.index(self.player_pos)
            if self.cargo[quest] == 0:
                self.started = True
                self.cargo[quest] = 1
                self.money += self.prepaid * self.rewards[quest] * 3
                return self.prepaid * self.rewards[quest] * 3

        elif self.player_pos in self.destinations:
            quest = self.destinations.index(self.player_pos)
            if self.cargo[quest] == 1:
                self.ended = True
                self.cargo[quest] = 0
                self.money += (1 - self.prepaid) * self.rewards[quest] * 3
                return (1 - self.prepaid) * self.rewards[quest] * 3

        if self.player_pos in self.gas_stations:
            cost = self.gas - self.gas_max
            self.has_tanked = True
            if abs(cost) > self.money:
                cost = self.money * -1
                self.money = 0
                self.gas -= cost
            else:
                self.money += cost
                self.gas = self.gas_max
            return cost

        return self.action_wait()

    def sample_move(self):
        return random.randint(0, len(self.action_space) - 1)

    def print_map(self):
        print("")
        print("")
        print("")

        for i in range(self.map_size):
            line = "     "
            for j in range(self.map_size):
                line += self.map[i][j] + " "
            print(line)

        print("")
        print("")
        print("")




