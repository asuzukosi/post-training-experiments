export default function AboutPage(){
    return (
        <>
        <h1>About</h1>
        <p>Hello there. <br/> How do you do? </p>
        </>
    )
}

/**
 * JSX lets you put markup into JavaScript. 
 * Curly braces let you “escape back” into JavaScript so that you can embed some variable from your code and display it to the user. 
 * For example, this will display user.name
 * 
 * You can also “escape into JavaScript” from JSX attributes, 
 * but you have to use curly braces instead of quotes.
 * For example, className="avatar" passes the "avatar" string as the CSS class, but src={user.imageUrl} 
 * reads the JavaScript user.imageUrl variable value, 
 * and then passes that value as the src attribute
 */