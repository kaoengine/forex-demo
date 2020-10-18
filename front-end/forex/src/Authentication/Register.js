import React from 'react'

class Register extends React.Component{
    state = {username:'',password:'',email:'',gender:'male',isRegister:false}

    onHandleRegister = (e) => {
        this.setState({
           [e.target.name] : e.target.value,
           isRegister: !this.state.isRegister
        })
    }

    onRegisterSubmit = (e) => {
        e.preventDefault();
        const {username, password,email,gender,isRegister} = this.state;
        this.props.addUser(username,password,email,gender,isRegister);
        this.setState({username:'',password:'',email:'',gender:'male',isRegister:false})
    }

    

    render(){
        const {username, password,email} = this.state
        return(
            <form onSubmit={this.onRegisterSubmit}>
                <div className="flex__container__items">
                    <label>UserName</label>
                    <input type="text" placeholder="username" name="username" value={username} onChange={this.onHandleRegister}/>
                </div>
                <div className="flex_container__items">
                    <label>password</label>
                    <input type="password" placeholder="password" name="password" value={password} onChange={this.onHandleRegister}/>
                </div>
                <div className="flex_container__items">
                    <label>Email</label>
                    <input type="email" placeholder="email" name="email" value={email} onChange={this.onHandleRegister}/>
                </div>
                <div className="flex_container__items">
                    <label>Gender</label>
                    <div defaultValue='default' onChange={(e) => this.setState({gender: e.target.value})}>
                        <input type='radio' name="gender" value="male" />Male
                        <input type='radio' name="gender" value="female" />Female
                        <input type='radio' name="gender" value="others" />Others 
                    </div>
                </div>
                <button type="submit">Submit</button>
            </form>
        
        );
    }
}

export default Register