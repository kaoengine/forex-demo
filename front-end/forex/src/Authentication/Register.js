import React from 'react'
import styled ,{css} from 'styled-components';

const Wrapper = styled.div`
    font-family: Arial, Helvetica, sans-serif;
    background: linear-gradient(to bottom, #f05053, #e1eec3);
    margin:0;
    padding:0;
    height: 100vh;
    color: #555;
    box-sizing: border-box;
`

const sharedStyled = css`
    background-color: #eee;
    height: 40px;
    border-radius: 5px;
    border: 1px solid #ddd;
    margin: 10px 0 20px 0;
    box-sizing: border-box;
`

const StyledFormWrapper = styled.div`
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100%;
    padding: 0 20px;
`

const StyledForm = styled.form`
    width: 100%;
    max-width: 700px;
    padding: 60px;
    background-color: #fff;
    border-radius: 10px;
    box-sizing: border-box;
    box-shadow: 0px 0px 20px 0px rgba(0,0,0,0.2);
`

const StyledInput = styled.input`
    display: block;
    width: 100%;
    ${sharedStyled}
`

const StyledFieldset = styled.fieldset`
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 10px;
    margin: 20px 0;
    
    legend{
        padding: 0 10px;
    }
    label{
        padding-right: 20px;
    }
    input{
        margin-right: 10px;
    }
`

const StyledButton = styled.input`
    display: block;
    background-color: #f7797d;
    color: #fff;
    font-size: .9rem;
    border:0;
    border-radius: 5px;
    height: 40px;
    padding: 0 20px;
    cursor: pointer;
    box-sizing: border-box;
    margin: 30px 0 20px 0;
`


class Register extends React.Component{
    constructor(props){
        super(props);
        this.state = { username:'',
                        password:'',
                        email:'',
                        gender:'',
                        isRegister:false,
                    }

    }

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
            <Wrapper>
                <StyledFormWrapper>
                    <StyledForm onSubmit={this.onRegisterSubmit}>
                        <h2>Register Form</h2>
                        <label><strong>UserName</strong></label>
                        <StyledInput type="text" name="username" value={username} onChange={this.onHandleRegister}/>
                        <label><strong>password</strong></label>
                        <StyledInput type="password" name="password" value={password} onChange={this.onHandleRegister}/>
                        <label><strong>Email</strong></label>
                        <StyledInput type="email" name="email" value={email} onChange={this.onHandleRegister}/>
                        <StyledFieldset className="flex_container__items">
                            <legend>Gender</legend>
                            <div defaultValue='default' onChange={(e) => this.setState({gender: e.target.value})}>
                                <label>
                                    <input type='radio' name="gender" value="male" />Male
                                </label>
                                <label>
                                    <input type='radio' name="gender" value="female" />Female
                                </label>
                                <label>
                                    <input type='radio' name="gender" value="others" />Others 
                                </label>
                            </div>
                        </StyledFieldset>
                        <StyledButton type="submit" value="Submit" />
                    </StyledForm>
                </StyledFormWrapper>
            </Wrapper>
        );
    }
}

export default Register