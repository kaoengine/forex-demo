import React, { Component } from "react";
import { Line } from "react-chartjs-2";
import {FormatDateHelper} from "../helpers/FormatDateHelper";
import {fetchAPI} from "../helpers/FetchAPI";

import moment from "moment";

export default class StockChart extends Component {
  constructor() {
    super();
    this.state = { data: [] };
  }

  render() {
    const {labels} = this.state.data
    return (
      <Line
        data={{
          labels: FormatDateHelper(labels),
          datasets:this.state.data.datasets,
        }}
        options={{
          title: {
            display: true,
            text: "World population per region (in millions)",
            fontSize: 25
          },
          legend: {
            display: true,
            position: "bottom",
          },
    
        }}
      />
    );
  }

  async componentDidMount() {
    const BASE_URL = 'http://localhost:6011/forex'
    try {
      console.log('test');
      const current_date_query= {"currDate":moment(new Date()).format("YYYY-MM-DD HH:mm:ss")}
      const response = await fetchAPI(BASE_URL);
      const json = await response.json();
      console.log(json);
      this.setState({ data: {labels: [1500, 1600, 1700, 1750, 1800, 1850, 1900, 1950, 1999, 2050], ...json} });
    } catch (error) {
      console.log(error);
    }
  }
}
