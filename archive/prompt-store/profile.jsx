const user = {
    name: 'Hedy Lamarr',
    imageUrl: 'https://i.imgur.com/yXOvdOSs.jpg',
    imageSize: 90,
};

export default function Profile(){
    let content;
    if (isLoggedIn){
        content = <AdminPanel />;
    }{
        content = <LoginForm />;
    }
    return (
        <>
            <h1>{user.name}</h1>
            <img className="avatar" src={user.imageUrl} alt={'Photo of ' + user.name} style={{
                width: user.imageSize,
                height: user.imageSize
            }} />
            {content}
            <div>{isLoggedIn? (<AdminPanel/>) : (<LoginForm/>)}</div>
        </>
    )
}