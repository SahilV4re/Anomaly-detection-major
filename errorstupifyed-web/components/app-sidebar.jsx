"use client";

import * as React from "react";
import {
  AudioWaveform,
  BookOpen,
  Bot,
  BotMessageSquare,
  ChartPie,
  Command,
  Frame,
  GalleryVerticalEnd,
  LayoutDashboard,
  LogOutIcon,
  Map,
  PieChart,
  Settings2,
  SquareTerminal,
  Text,
} from "lucide-react";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Cookies from "js-cookie";
import { NavMain } from "@/components/nav-main";
import { NavUser } from "@/components/nav-user";
import axios from "axios";
import { TeamSwitcher } from "@/components/team-switcher";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarRail,
} from "@/components/ui/sidebar";

// This is sample data.
var data = {
  user: {
    name: "Meet Chavan",
    email: "meetchavan@gmail.com",
    avatar: "/avatars/shadcn.jpg",
  },

  navMain: [
    {
      title: "Dashboard",
      url: "/dashboard",

      icon: LayoutDashboard,
      isActive: true,
    },
    {
      title: "Audits",
      url: "/audit",
      icon: ChartPie,
      isActive: false,
    },
    // {
    //   title: "Records",
    //   url: "#",
    //   icon: Text,
    //   isActive: false,
    // },


    {
      title: "Chat",
      url: "/chat",
      icon: BotMessageSquare,
      isActive: false,
    }, {
      title: "Preferences",
      url: "/settings",
      icon: Settings2,
      isActive: false,
    },

    {
      title: "Logout",
      url: "/",
      icon: LogOutIcon,
      isActive: false,
    },
  ],
};

export function AppSidebar({ ...props }) {
  const router = useRouter();
  const SERVER_URL = process.env.NEXT_PUBLIC_API_URL;
  const [user_data, setData] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const uid = Cookies.get("uid"); // Replace with dynamic UID if needed
        const response = await axios.get(`${SERVER_URL}/get_user/${uid}`);

        setData(response.data);
        data['user']['name'] = response.data.data.name;
        data['user']['email'] = response.data.data.email;

        console.log("User data fetched:", response.data);
      } catch (err) { }
    };

    fetchData();
  }, []);

  return (
    <Sidebar collapsible="icon" {...props}>
      <SidebarHeader>
        <TeamSwitcher teams={data.teams} />
      </SidebarHeader>
      <SidebarContent>
        <NavMain items={data.navMain} />
      </SidebarContent>

      <SidebarFooter>
        <NavUser user={data.user} />
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}
