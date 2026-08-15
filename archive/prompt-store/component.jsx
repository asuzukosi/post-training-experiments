/*
React apps are made of components, 
A component is a piece of the UI that has its onwl logic and appearance. 
A component can ba as small a button or as large as an entire page

functions that start with 'use' are called hooks in react, for example the
useState is a hook in react
*/

import {useState} from 'react';

function MyButton(){
    const [count, setCount] = useState(0);
    function handleClick(){
        setCount(count + 1) ;
        alert("You clicked me!");
    }
    return (
        <button onClick={handleClick}>
            I am a button you have clicked me {count} times.
        </button>
    )
}

export default MyButton