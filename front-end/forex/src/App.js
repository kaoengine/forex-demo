import React from 'react';
import  StockChart  from "./Components/StockChart";
import Authenticate  from "./Authentication/Authenticate";
import Login from "./Authentication/Register";
import Protected from "./Authentication/Protected";
import {BrowserRouter as Router, Switch, Route} from "react-router-dom";

const App = () => {
  return (
      <Router>
        <Switch>
          <Route path="/" exact component={Authenticate}></Route>
          <Route path="/protected" component={Protected}></Route>
          <Route path="/login" component={Login}></Route>
          <Route path="/chart" component={StockChart}></Route>
        </Switch>
    </Router>
  );
}

export default App;
