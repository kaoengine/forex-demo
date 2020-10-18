import React, { Fragment } from 'react'
import Register from "./Register";
import { Redirect } from 'react-router-dom'

class Authenticate extends React.Component{
    constructor(props){
        super(props);
        this.state={ Users: this.getUser() }
    }

    addUser = (name,password,email,gender,isRegister) => {
        this.setState({
            Users: [...this.state.Users,
                {   username: name,
                password: password, 
                email: email,
                gender:gender,
                isRegister:isRegister
                }
            ],
        })
    }

    saveLocalStorageUser = () => {
        localStorage.setItem("users", JSON.stringify(this.state.Users));
    }

    getUser = () => {
        const users = localStorage.getItem('users');
        return users ? JSON.parse(users) : []
    }

    componentDidUpdate(){
        this.saveLocalStorageUser()
    }

    render(){
        const user = localStorage.getItem('users');
        if(user) return <Redirect to="/protected"/>
        return(
            <Fragment>
                <Register addUser={this.addUser}/>
            </Fragment>
        )
    }
}

export default Authenticate