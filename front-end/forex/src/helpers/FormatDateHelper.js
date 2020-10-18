import moment from "moment";

export const FormatDateHelper = (arr) => {
    return arr ? arr.map(ele => moment(new Date(ele)).format("YYYY-MM-DD HH:mm:ss")) : null
}