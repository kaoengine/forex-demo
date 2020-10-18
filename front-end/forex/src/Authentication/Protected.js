import React from 'react'
import {Link} from "react-router-dom"

const Protected = () => {
    return (
        <div>
            <h1>You have Successfully Registered!!</h1>
            <h4>Please check your Email to verify!</h4>
            <Link to="/chart">GO TO CHART</Link>
        </div>
    )
}

export default Protected