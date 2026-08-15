
/*
Its important to add keys to your data wehn entering a list
React uses it for reordering, deleting and inserting
*/
export default function Products(){
    const products = [
        { title: 'Cabbage', id: 1 },
        { title: 'Garlic', id: 2 },
        { title: 'Apple', id: 3 },
    ];

    const listItems = products.map(product => 
    <li key={product.id}
        style={{
            color: product.title === 'Apple' ? 'magenta' : 'darkgreen'
        }}>
        {product.title}
    </li>)
    return (
        <ul>
        {listItems}
        </ul>
    )
}