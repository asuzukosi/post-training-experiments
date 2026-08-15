import React from 'react';
import './App.css';
import { useState } from 'react';

function ProductCategoryRow({category}:any){
  return (
    <tr>
      <th colSpan={2}>
        {category}
      </th>
    </tr>
  )
}

function ProductRow({product}: any) {
  const name = product.stocked ? product.name:
  <span style={{color: 'red'}}>
    {product.name}
  </span>

  return (
    <tr>
      <td>
        {name}
      </td>
      <td>
        {product.price}
      </td>
    </tr>
  );
}


function ProductTable({products, filterText, inStockOnly}: any){
  const rows: any = [];
  let lastCategory: any = null;

  products.forEach((product: any) => {
    if(product.name.toLowerCase().indexOf(filterText.toLowerCase()) === -1){
      return;
    }
    if(inStockOnly && !product.stocked){
      return;
    }
    if(product.category !== lastCategory) {
      rows.push(
        <ProductCategoryRow 
          category={product.category}
          key={product.category}
        />
      );
  }
  rows.push(
    <ProductRow 
      product={product}
      key={product.name}
    />
  );
  lastCategory = product.category;
});

  return (
    <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Price</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
  )
}


function SearchBar({filterText, inStockOnly, onFilterTextChange, onIsStockOnlyChange}: any){
  return (
    <form>
        <input type="text" value={filterText} placeholder='Search...' onChange={(e)=> onFilterTextChange(e.target.value)} />
        <label>
          <input type='checkbox' onChange={()=>onIsStockOnlyChange(!inStockOnly)}/>
          {' '}
          Only show products in stock
        </label>
    </form>
  );
}

interface FilterableProductTableProps {
  products: any
}
function FilterableProductTable({products}: FilterableProductTableProps){
  
  const [filterText, setFilterText] = useState('');
  const [inStockOnly, setInStockOnly] = useState(false);
  return (
    <div>
      <SearchBar filterText = {filterText} 
                 inStockOnly={inStockOnly}
                 onFilterTextChange={setFilterText}
                 onIsStockOnlyChange={setInStockOnly}/>
      <ProductTable products={products} filterText = {filterText} inStockOnly={inStockOnly}/>
    </div>
  )
}

const PRODUCTS = [
  {category: "Fruits", price: "$1", stocked: true, name: "Apple"},
  {category: "Fruits", price: "$1", stocked: true, name: "Dragonfruit"},
  {category: "Fruits", price: "$2", stocked: false, name: "Passionfruit"},
  {category: "Vegetables", price: "$2", stocked: true, name: "Spinach"},
  {category: "Vegetables", price: "$4", stocked: false, name: "Pumpkin"},
  {category: "Vegetables", price: "$1", stocked: true, name: "Peas"}
];

function App() {
  return (
    <div>
      <FilterableProductTable products={PRODUCTS} />  
    </div>
  );
}

export default App;
