from flask import Flask, render_template, send_from_directory, request, make_response
import chess
import numpy as np
import operator
import torch
import json
from model.train import ChessReward

app = Flask(__name__)

# load and create model
model = torch.load("dqn_chess.pth")

json_file = open('generalized_moves.json')
json_str = json_file.read()
generalized_moves = json.loads(json_str)
state_board = np.zeros((1, 65))
switch = ChessReward().rewards
squares = chess.SQUARES


@app.route('/')
def index():
    return render_template("reinforced.html")


@app.route('/minimax')
def minimax():
    return render_template('minimax-alpha-beta.html')


@app.route('/random')
def random():
    return render_template('random.html')


@app.route('/img/<path:path>')
def send_js(path):
    return send_from_directory('static/img', path)


@app.route('/send_move', methods=['GET', 'POST'])
def sender_moves():
    if request.method == "POST":
        print("request : " + str(request.json))
        fen = request.json['fen']
        turn = request.json['turn']
        board = chess.Board(fen=fen)
        print("turn : " + str(turn))
        if str(turn) != "b":
            board.turn = True
        else:
            board.turn = False
        Q = {}
        if board.turn:
            mult = 1
        else:
            mult = -1
        a = 0
        for i in squares:
            state_board[0][a] = mult * switch[str(board.piece_at(i))]
            a += 1
        for i in board.legal_moves:
            state_board[0][64] = generalized_moves[str(i)]
            with torch.no_grad():
                Q[str(i)] = model(state_board)
        best_move = max(Q.items(), key=operator.itemgetter(1))[0]
        print("Best move is : " + str(best_move))

        return json.dumps(best_move), 200, {'ContentType': 'application/json'}


if __name__ == '__main__':
    app.run()
