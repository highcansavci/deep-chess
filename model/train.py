import torch
import torch.nn as nn
import chess
import operator
import numpy as np
import os
import json
from config.config import Config
from dataset.chess_dataset import create_dataloader, ChessDataset


class ChessPolicy(nn.Module):
    def __init__(self):
        super(ChessPolicy, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(in_features=65, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=18),
            nn.ReLU(),
            nn.Linear(in_features=18, out_features=18),
            nn.ReLU(),
            nn.Linear(in_features=18, out_features=18),
            nn.ReLU(),
            nn.Linear(in_features=18, out_features=18),
            nn.ReLU(),
            nn.Linear(in_features=18, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1),
            nn.ReLU(),
        )

    def forward(self, board_state):
        return self.policy(board_state)


class ChessReward(object):
    def __init__(self):
        self.rewards = {
            "p": 10,
            "P": -10,
            "q": 100,
            "Q": -100,
            "n": 25,
            "N": -25,
            "r": 50,
            "R": -50,
            "b": 30,
            "B": -30,
            "k": 1000,
            "K": -1000,
            "None": 0
        }

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ChessReward, cls).__new__(cls)
        return cls.instance


def evaluate_board(turn):
    lookup = chess.SQUARES
    total, board_idx = 0, 0
    mult = 1 if turn else -1
    for lkp in lookup:
        reward_ = mult * chess_reward[str(board.piece_at(lkp))]
        total += reward_
        q_board[0][board_idx] = reward_
        board_idx += 1
    return total


def get_int(move):
    try:
        return general_moves[str(move)]
    except:
        general_moves[str(move)] = len(general_moves)
        return general_moves[str(move)]


def reward(fen_history, moves, lose_fen, lose_moves):
    inputs = []
    targets = []
    reward_loss = 0

    for idx in range(len(fen_history)):
        gamma_ = 1 / len(fen_history)
        fen_history[idx][0][64] = get_int(moves[idx])
        inputs.append(fen_history[idx])
        targets.append(model(torch.tensor(np.array(fen_history[idx]), dtype=torch.float)) + winner_reward * (gamma_ * idx))

    for idx in range(len(lose_fen)):
        gamma_ = 1 / len(lose_fen)
        fen_history[idx][0][64] = get_int(lose_moves[idx])
        inputs.append(lose_fen[idx])
        targets.append(model(torch.tensor(np.array(lose_fen[idx]), dtype=torch.float)) + loser_malus * (gamma_ * idx))
        idx += 1

    data_loader = create_dataloader(ChessDataset(inputs, targets))

    for data in data_loader:
        optimizer.zero_grad()
        inputs_, targets_ = data
        output = model(inputs_)
        loss = criterion(output, targets_)

        loss.backward()
        optimizer.step()
        reward_loss += loss.item()


if __name__ == "__main__":
    q_board = np.zeros((1, 65))
    config = Config().config
    chess_reward = ChessReward().rewards
    arguments = {'training_games': config["model"]["number_of_games"],
                 'winner_reward': config["model"]["winner_reward"],
                 'loser_malus': config["model"]["loser_malus"], 'epsilon': config["model"]["epsilon"],
                 'decremental_epsilon': config["model"]["decremental_epsilon"], 'gamma': config["model"]["gamma"]}
    general_moves = {}

    steps = 1000
    training_games = int(arguments['training_games'])
    winner_reward = int(arguments['winner_reward'])
    loser_malus = int(arguments['loser_malus'])
    epsilon = float(arguments['epsilon'])
    decremental_epsilon = float(arguments['decremental_epsilon'])
    gamma = float(arguments['gamma'])

    print("Training the Deep-Q-Network with parameters : ")
    print("Number of training games : " + str(training_games))
    print("Winner Reward : " + str(winner_reward))
    print("Loser Malus : " + str(loser_malus))
    print("Epsilon : " + str(epsilon))
    print("Decremental Epsilon : " + str(decremental_epsilon))
    print("Gamma : " + str(gamma))

    winners = {}  # Variable for counting number of wins of each player
    model = ChessPolicy().to(config["model"]["device"])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])

    for joum in range(0, steps):
        i = 0
        board = chess.Board()
        epsilon = 1
        decremental_epsilon = 1 / training_games
        for i in range(training_games):
            os.system('clear')
            print("/------------------ Training -----------------/")
            print("Step (" + str(joum) + "/" + str(steps) + ")")
            print("Game N°" + str(i))
            print("WINNERS COUNT : \n" + str(winners))
            print("Number of remaining training games : " + str(training_games - i))
            print("Winner Reward : " + str(winner_reward))
            print("Loser Malus : " + str(loser_malus))
            print("Epsilon : " + str(epsilon))
            print("Decremental Epsilon : " + str(decremental_epsilon))
            print("Gamma : " + str(gamma))
            black_moves = []
            white_moves = []
            black_fen_history = []
            white_fen_history = []
            number_of_moves = 0

            while not board.is_game_over():
                number_of_moves += 1
                prob = np.random.rand()
                if prob <= epsilon:
                    move_list = list(board.legal_moves)
                    n_move = np.random.randint(0, len(move_list))
                    move = str(move_list[n_move])
                else:
                    evaluate_board(True)
                    Q = {}
                    for kr in board.legal_moves:
                        br = get_int(kr)
                        q_board[0][64] = br
                        Q[kr] = model(torch.tensor(q_board, dtype=torch.float))
                    move = max(Q.items(), key=operator.itemgetter(1))[0]
                base_eval = evaluate_board(board.turn)
                fen = str(board.fen())
                if board.turn:
                    white_moves.append(move)
                    white_fen_history.append(np.array(q_board, copy=True))
                else:
                    black_moves.append(move)
                    black_fen_history.append(np.array(q_board, copy=True))
                board.push(chess.Move.from_uci(str(move)))

            if board.result() == "1-0":
                reward(white_fen_history, white_moves, black_fen_history, black_moves)
            elif board.result() == "0-1":
                reward(black_fen_history, black_moves, white_fen_history, white_moves)
            try:
                winners[str(board.result())] = winners[str(board.result())] + 1
            except:
                winners[str(board.result())] = 1
            board.reset()
            epsilon -= decremental_epsilon

    torch.save(model, "dqn_chess.pth")
    print("WINNERS COUNT : \n" + str(winners))
    with open('generalized_moves.json', 'w') as fp:
        json.dump(general_moves, fp)
