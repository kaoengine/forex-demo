import React, { Component } from "react";
import { Line } from "react-chartjs-2";
export default class StockChart extends Component {
  constructor() {
    super();
    this.state = { data: [] };
  }
  render() {
    return (
      <Line
        data={{
          labels: ['12:15', '12:30', '13:00', '13:15', '13:30', '13:45', '14:00', '14:15', '14:30', '14:45'],
          datasets:this.state.data.datasets,
        }}
        options={{
          title: {
            display: true,
            text: "World population per region (in millions)",
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
    try {
      console.log('test');
      const response = await fetch(`http://localhost:6011/forex`);
      const json = await response.json();
      console.log(json);
      this.setState({ data: {labels: [1500, 1600, 1700, 1750, 1800, 1850, 1900, 1950, 1999, 2050], ...json} });
    } catch (error) {
      console.log(error);
    }
  }
}
