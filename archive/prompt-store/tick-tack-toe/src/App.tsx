import React from 'react';
import './App.css';
import { useState } from 'react';

/*
* To remember things in react we use useState
*/

interface SquareProps {
  value: string | null;
  onClick: any
}

function Square({value, onClick}: SquareProps) {

  return <button className="square" onClick={onClick}>{value}</button>
}


interface BoardProps {
  xIsNext: any;
  squares: any;
  onPlay: any;
}

function Board({xIsNext, squares, onPlay}: BoardProps) {

  function calculateWinner(squares: null[] | string[]): null | string {
    const lines = [
      [0, 1, 2],
      [3, 4, 5],
      [6, 7, 8],
      [0, 3, 6],
      [1, 4, 7],
      [2, 5, 8],
      [0, 4, 8],
      [2, 4, 6]
    ];

    for(let i = 0; i < lines.length; i++) {
      const [a, b, c] = lines[i]
      if(squares[a] && squares[a] === squares[b] && squares[a] === squares[c])
        return squares[a];
    }

    return null;
  }

  const winner = calculateWinner(squares);

  let status;

  if(winner){
    status = "Winner: " + winner;
  }else{
    status  = "Next player: "+ (xIsNext? "X" : "O")
  }
  function handleClick(i: number){
    if(squares[i] || calculateWinner(squares)){
      return
    }
    const nextSquares = squares.slice();
    if(xIsNext){
      nextSquares[i] = "X";
    }else{
      nextSquares[i] = "O"
    }
    onPlay(nextSquares)

  }

  return (
    <div className="App">
      <div className="status"><h6>{status}</h6></div>
      <div className="board-row">
        <Square value={squares?.[0]} onClick={()=>{handleClick(0)}}/>
        <Square value={squares?.[1]} onClick={()=>{handleClick(1)}}/>
        <Square value={squares?.[2]} onClick={()=>{handleClick(2)}}/>
      </div>
     
      <div className="board-row">
        <Square value={squares?.[3]} onClick={()=>{handleClick(3)}}/>
        <Square value={squares?.[4]} onClick={()=>{handleClick(4)}}/>
        <Square value={squares?.[5]} onClick={()=>{handleClick(5)}}/>
      </div>

      <div className="board-row">
        <Square value={squares?.[6]} onClick={()=>{handleClick(6)}}/>
        <Square value={squares?.[7]} onClick={()=>{handleClick(7)}}/>
        <Square value={squares?.[8]} onClick={()=>{handleClick(8)}}/>
      </div>

    </div>
  );
}

function App(){

  const [history, setHistory] = useState([Array(9).fill(null)])
  const [currentMove, setCurrentMove] = useState(0)
  const currentSquares = history[currentMove];

  const xIsNext = currentMove % 2 === 0

  function handlePlay(nextSquares: string[] | null[]){
    const nextHistory = [...history.slice(0, currentMove+1), nextSquares];
    setHistory(nextHistory);
    setCurrentMove(nextHistory.length - 1);
  }

  function jumpTo(nextMove: any){
    setCurrentMove(nextMove);
  }

  const moves = history.map((squares, move) => {
    let description;
    if (move > 0){
      description = 'Go to move #' + move;
    } else {
      description = 'Go to game start';
    }

    return (
      <li>
        <button key={move} onClick={() => jumpTo(move)}>{description}</button>
      </li>
    );
  });

  return (
    <div className='game'>
      <div className='game-board'>
        <Board xIsNext={xIsNext} squares={currentSquares} onPlay={handlePlay}/>
      </div>
      <div className='game-info'>
        <ol>{moves}</ol>
      </div>
    </div>

  )
}

export default App;
/**
 * Why immuatability is important in react
 * There are two approaches to changing data, the first is to directly mutate the data values
 * the second approach is to replace the data with a new copy which has the desired changes
 * Immuatability allows us to be able to time travel through our state
 * Immutability also means that all other related data is updated, which means that 
 * all related components are also rerendered
 */
