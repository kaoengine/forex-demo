import React from 'react'
import {Link} from "react-router-dom"
import styled from "styled-components";

const ProtectedWrapper = styled.div`
    display: flex;
    background: linear-gradient(to bottom, #f05053, #e1eec3);
    flex-direction: column;
    height: 100vh;
    color: #fff;
    justify-content: center;
    align-items: center;
    font-size: 1.75em;
`
const Protected = () => {
    return (
        <ProtectedWrapper>
            <h1>You have Successfully Registered!!</h1>
            <h4>Please check your Email to verify!</h4>
            <Link to="/chart">GO TO CHART</Link>
        </ProtectedWrapper>
    )
}

export default Protected