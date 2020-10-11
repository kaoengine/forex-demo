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
          labels: [1500, 1600, 1700, 1750, 1800, 1850, 1900, 1950, 1999, 2050],
          datasets: [
            {
              data: [86, 114, 106, 106, 107, 111, 133, 221, 783, 2478],
              label: "Africa",
              borderColor: "#3e95cd",
              fill: false,
            },
            {
              data: [282, 350, 411, 502, 635, 809, 947, 1402, 3700, 5267],
              label: "Asia",
              borderColor: "#8e5ea2",
              fill: false,
            },
            {
              data: [168, 170, 178, 190, 203, 276, 408, 547, 675, 734],
              label: "Europe",
              borderColor: "#3cba9f",
              fill: false,
            },
            {
              data: [40, 20, 10, 16, 24, 38, 74, 167, 508, 784],
              label: "Latin America",
              borderColor: "#e8c3b9",
              fill: false,
            },
            {
              data: [6, 3, 2, 2, 7, 26, 82, 172, 312, 433],
              label: "North America",
              borderColor: "#c45850",
              fill: false,
            },
          ],
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
      const response = await fetch(`http://127.0.0.1:6011/how are you`);
      const json = await response.json();
      console.log(json);
      // this.setState({ data: json });
    } catch (error) {
      console.log(error);
    }
  }
}
